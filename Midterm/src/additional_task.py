import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math

class FeedForwardNetwork:
    def __init__(self, hidden_layers, rng=None):
        """
        hidden_layers: список кількостей нейронів у прихованих шарах, напр. [4, 6, 3]
        Вхід: 1 нейрон (x), вихід: 1 нейрон (y)
        """
        self.hidden_layers = [int(n) for n in hidden_layers if int(n) > 0]
        self.layer_sizes = [1] + self.hidden_layers + [1]
        self.rng = rng or np.random.default_rng()
        self._init_params()

    def _init_params(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            W = self.rng.normal(0.0, 0.5, size=(n_in, n_out))
            b = np.zeros(n_out)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        X: (n_samples, 1)
        """
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i < len(self.weights) - 1:
                a = np.tanh(z)
            else:
                a = z
        return a

    def train(self, X, y, epochs=200, lr=0.01):
        """
        Проста повна градієнтна спускова оптимізація MSE.
        """
        n_samples = X.shape[0]
        for _ in range(epochs):
            activations = [X]
            zs = []
            a = X
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                z = a @ W + b
                zs.append(z)
                if i < len(self.weights) - 1:
                    a = np.tanh(z)
                else:
                    a = z
                activations.append(a)

            y_pred = activations[-1]
            dLoss_da = 2.0 * (y_pred - y) / n_samples

            delta = dLoss_da
            for layer_idx in reversed(range(len(self.weights))):
                a_prev = activations[layer_idx]
                z = zs[layer_idx]

                dW = a_prev.T @ delta
                db = np.sum(delta, axis=0)

                self.weights[layer_idx] -= lr * dW
                self.biases[layer_idx] -= lr * db

                if layer_idx > 0:
                    delta = delta @ self.weights[layer_idx].T
                    # похідна tanh: 1 - a^2
                    a_prev_activated = activations[layer_idx]
                    delta *= (1.0 - a_prev_activated ** 2)

    def mse(self, X, y):
        y_pred = self.forward(X)
        return float(np.mean((y_pred - y) ** 2))


class StructuralGA:
    def __init__(self, x_train, y_train, max_neurons_total, target_mse,
                 max_layers=4, pop_size=20, generations=30,
                 crossover_rate=0.8, mutation_rate=0.3, rng=None,
                 progress_callback=None):
        """
        Хромосома: список довжини max_layers, кожен ген – кількість нейронів у шарі (0..max_neurons_total).
        Якщо сума нейронів > max_neurons_total, архітектура штрафується.
        """
        self.x_train = x_train.reshape(-1, 1)
        self.y_train = y_train.reshape(-1, 1)
        self.max_neurons_total = int(max_neurons_total)
        self.target_mse = float(target_mse)
        self.max_layers = max_layers
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = rng or np.random.default_rng()
        self.progress_callback = progress_callback or (lambda *args, **kwargs: None)

        self.population = self._init_population()

    def _random_architecture(self):
        genes = self.rng.integers(0, self.max_neurons_total + 1, size=self.max_layers)
        if np.sum(genes) == 0:
            layer = self.rng.integers(0, self.max_layers)
            genes[layer] = self.rng.integers(1, self.max_neurons_total + 1)
        return genes

    def _init_population(self):
        return np.array([self._random_architecture() for _ in range(self.pop_size)], dtype=int)

    def _architecture_to_list(self, genes):
        return [int(n) for n in genes if int(n) > 0]

    def _evaluate_individual(self, genes):
        total_neurons = int(np.sum(genes))
        if total_neurons == 0 or total_neurons > self.max_neurons_total:
            return 1e6

        hidden_layers = self._architecture_to_list(genes)
        net = FeedForwardNetwork(hidden_layers, rng=self.rng)
        net.train(self.x_train, self.y_train, epochs=200, lr=0.01)
        return net.mse(self.x_train, self.y_train)

    def _evaluate_population(self):
        errors = np.zeros(self.pop_size)
        for i, genes in enumerate(self.population):
            errors[i] = self._evaluate_individual(genes)
        return errors

    def _tournament_selection(self, errors, k=3):
        idx = self.rng.integers(0, self.pop_size, size=k)
        best_idx = idx[np.argmin(errors[idx])]
        return self.population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        if self.rng.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = self.rng.integers(1, self.max_layers)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, genes):
        for i in range(self.max_layers):
            if self.rng.random() < self.mutation_rate:
                delta = self.rng.integers(-2, 3)  # -2..+2
                genes[i] = np.clip(genes[i] + delta, 0, self.max_neurons_total)
        if np.sum(genes) == 0:
            layer = self.rng.integers(0, self.max_layers)
            genes[layer] = self.rng.integers(1, self.max_neurons_total + 1)
        return genes

    def run(self):
        best_overall_genes = None
        best_overall_error = float("inf")

        for gen in range(self.generations):
            errors = self._evaluate_population()
            best_idx = int(np.argmin(errors))
            best_error = float(errors[best_idx])
            best_genes = self.population[best_idx].copy()

            if best_error < best_overall_error:
                best_overall_error = best_error
                best_overall_genes = best_genes.copy()

            arch_str = "-".join(str(n) for n in self._architecture_to_list(best_genes))
            if arch_str == "":
                arch_str = "без прихованих шарів"
            self.progress_callback(gen, best_error, arch_str, list(best_genes))

            if best_error <= self.target_mse:
                break

            new_pop = []
            elite_count = max(1, self.pop_size // 5)
            elite_indices = np.argsort(errors)[:elite_count]
            for idx in elite_indices:
                new_pop.append(self.population[int(idx)].copy())

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(errors)
                p2 = self._tournament_selection(errors)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    c2 = self._mutate(c2)
                    new_pop.append(c2)

            self.population = np.array(new_pop, dtype=int)

        final_hidden_layers = self._architecture_to_list(best_overall_genes)
        final_net = FeedForwardNetwork(final_hidden_layers, rng=self.rng)
        final_net.train(self.x_train, self.y_train, epochs=300, lr=0.01)
        final_mse = final_net.mse(self.x_train, self.y_train)
        return final_hidden_layers, final_mse


class App:
    def __init__(self, root):
        self.root = root
        root.title("Структурний синтез НМ за допомогою ГА")

        main_frame = ttk.Frame(root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        ttk.Label(main_frame, text="Функція f(x):").grid(row=0, column=0, sticky="w")
        self.func_entry = ttk.Entry(main_frame, width=40)
        self.func_entry.insert(0, "np.cos(np.sin(x)) * np.sin(x)")
        self.func_entry.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(main_frame, text="Інтервал x від:").grid(row=1, column=0, sticky="w")
        self.x_from_entry = ttk.Entry(main_frame, width=10)
        self.x_from_entry.insert(0, "0.0")
        self.x_from_entry.grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(main_frame, text="до:").grid(row=1, column=2, sticky="w")
        self.x_to_entry = ttk.Entry(main_frame, width=10)
        self.x_to_entry.insert(0, "5.0")
        self.x_to_entry.grid(row=1, column=3, sticky="w", pady=2)

        ttk.Label(main_frame, text="Кількість навчальних точок:").grid(row=2, column=0, sticky="w")
        self.n_points_entry = ttk.Entry(main_frame, width=10)
        self.n_points_entry.insert(0, "50")
        self.n_points_entry.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(main_frame, text="Максимальна кількість нейронів (всього):").grid(row=3, column=0, sticky="w")
        self.max_neurons_entry = ttk.Entry(main_frame, width=10)
        self.max_neurons_entry.insert(0, "20")
        self.max_neurons_entry.grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(main_frame, text="Максимальна похибка (MSE):").grid(row=4, column=0, sticky="w")
        self.target_mse_entry = ttk.Entry(main_frame, width=10)
        self.target_mse_entry.insert(0, "0.01")
        self.target_mse_entry.grid(row=4, column=1, sticky="w", pady=2)

        ga_label = ttk.LabelFrame(main_frame, text="Параметри генетичного алгоритму", padding=5)
        ga_label.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(10, 5))

        ttk.Label(ga_label, text="Розмір популяції:").grid(row=0, column=0, sticky="w")
        self.pop_size_entry = ttk.Entry(ga_label, width=10)
        self.pop_size_entry.insert(0, "20")
        self.pop_size_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(ga_label, text="Кількість поколінь (макс):").grid(row=0, column=2, sticky="w")
        self.generations_entry = ttk.Entry(ga_label, width=10)
        self.generations_entry.insert(0, "30")
        self.generations_entry.grid(row=0, column=3, sticky="w")

        ttk.Label(ga_label, text="Кількість прихованих шарів (макс):").grid(row=1, column=0, sticky="w")
        self.max_layers_entry = ttk.Entry(ga_label, width=10)
        self.max_layers_entry.insert(0, "4")
        self.max_layers_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(ga_label, text="Ймовірність кросоверу:").grid(row=1, column=2, sticky="w")
        self.crossover_entry = ttk.Entry(ga_label, width=10)
        self.crossover_entry.insert(0, "0.8")
        self.crossover_entry.grid(row=1, column=3, sticky="w")

        ttk.Label(ga_label, text="Ймовірність мутації гена:").grid(row=2, column=0, sticky="w")
        self.mutation_entry = ttk.Entry(ga_label, width=10)
        self.mutation_entry.insert(0, "0.3")
        self.mutation_entry.grid(row=2, column=1, sticky="w")

        self.start_button = ttk.Button(main_frame, text="Запустити пошук", command=self.start_search)
        self.start_button.grid(row=6, column=0, columnspan=4, pady=(10, 5))

        self.result_label = ttk.Label(main_frame, text="Результат: поки що немає", foreground="blue")
        self.result_label.grid(row=7, column=0, columnspan=4, sticky="w", pady=(5, 0))

        for i in range(4):
            main_frame.columnconfigure(i, weight=1)

        self.progress_window = None
        self.progress_text = None

    def _open_progress_window(self):
        if self.progress_window is not None and tk.Toplevel.winfo_exists(self.progress_window):
            return
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Хід роботи генетичного алгоритму")

        frame = ttk.Frame(self.progress_window, padding=5)
        frame.pack(fill="both", expand=True)

        self.progress_text = tk.Text(frame, width=80, height=25)
        self.progress_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.progress_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.progress_text.configure(yscrollcommand=scrollbar.set)

        self.progress_text.insert("end", "Покол.\tMSE\t\tАрхітектура (гени) / приховані шари\n")
        self.progress_text.insert("end", "-" * 70 + "\n")

    def _log_progress(self, gen, mse, arch_str, genes):
        if self.progress_text is None:
            return
        line = f"{gen:4d}\t{mse:.6f}\t{genes} / {arch_str}\n"
        self.progress_text.insert("end", line)
        self.progress_text.see("end")
        self.progress_window.update_idletasks()

    def _parse_function(self, expr, x):
        """
        Обчислює f(x) із введеного рядка expr.
        Дозволені: np, math, sin, cos, tan, exp, log, sqrt, pi, e.
        """
        local_dict = {
            "np": np,
            "math": math,
            "x": x,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "pi": math.pi,
            "e": math.e,
        }
        return eval(expr, {"__builtins__": {}}, local_dict)

    def start_search(self):
        try:
            expr = self.func_entry.get().strip()
            x_from = float(self.x_from_entry.get().strip())
            x_to = float(self.x_to_entry.get().strip())
            n_points = int(self.n_points_entry.get().strip())
            max_neurons = int(self.max_neurons_entry.get().strip())
            target_mse = float(self.target_mse_entry.get().strip())

            pop_size = int(self.pop_size_entry.get().strip())
            generations = int(self.generations_entry.get().strip())
            max_layers = int(self.max_layers_entry.get().strip())
            crossover_rate = float(self.crossover_entry.get().strip())
            mutation_rate = float(self.mutation_entry.get().strip())
        except ValueError:
            messagebox.showerror("Помилка вводу", "Перевірте правильність всіх числових полів.")
            return

        if n_points <= 1 or x_from >= x_to:
            messagebox.showerror("Помилка вводу", "Невірний інтервал або кількість точок.")
            return

        x_train = np.linspace(x_from, x_to, n_points)
        try:
            y_train = self._parse_function(expr, x_train)
        except Exception as e:
            messagebox.showerror("Помилка у виразі функції", f"Не вдалося обчислити f(x):\n{e}")
            return

        y_train = np.array(y_train, dtype=float)
        if y_train.shape != x_train.shape:
            messagebox.showerror("Помилка", "Функція повинна повертати значення того ж розміру, що й x.")
            return

        self._open_progress_window()
        self.progress_text.delete("1.0", "end")
        self.progress_text.insert("end", "Покол.\tMSE\t\tАрхітектура (гени) / приховані шари\n")
        self.progress_text.insert("end", "-" * 70 + "\n")

        self.start_button.config(state="disabled")
        self.result_label.config(text="Результат: пошук триває...", foreground="black")
        self.root.update_idletasks()

        rng = np.random.default_rng(42)

        ga = StructuralGA(
            x_train=x_train,
            y_train=y_train,
            max_neurons_total=max_neurons,
            target_mse=target_mse,
            max_layers=max_layers,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            rng=rng,
            progress_callback=self._log_progress,
        )

        hidden_layers, final_mse = ga.run()
        arch_str_final = "без прихованих шарів" if not hidden_layers else "-".join(str(n) for n in hidden_layers)

        self.result_label.config(
            text=f"Результат: архітектура 1-{arch_str_final}-1, MSE = {final_mse:.6f}",
            foreground="blue",
        )
        self.start_button.config(state="normal")

        messagebox.showinfo(
            "Готово",
            f"Пошук завершено.\n\nАрхітектура: 1-{arch_str_final}-1\nФінальна MSE: {final_mse:.6f}",
        )


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
