import numpy as np


def target_function(x, y):
    return np.cos(np.sin(y)) * np.sin(x)


class FeedForwardNN:
    def __init__(self, layer_sizes, activation=np.tanh):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.n_params = self._count_params()
        self.weights = []
        self.biases = []

    def _count_params(self):
        """
        Рахує загальну кількість параметрів:
        сума по шарах (n_in * n_out + n_out) – ваги + зсуви.
        """
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            total += n_in * n_out + n_out
        return total

    def decode_chromosome(self, chromosome):
        """
        Розпаковка хромосоми (вектор дійсних чисел) у матриці ваг та вектори зсувів.
        """
        if chromosome.shape[0] != self.n_params:
            raise ValueError(f"Очікується {self.n_params} параметрів, отримано {chromosome.shape[0]}")
        self.weights = []
        self.biases = []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            w_size = n_in * n_out

            W = chromosome[idx:idx + w_size].reshape(n_in, n_out)
            idx += w_size

            b = chromosome[idx:idx + n_out]
            idx += n_out

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Пряме поширення.
        X: масив форми (n_samples, n_features)
        Повертає: масив виходів (n_samples, 1)
        """
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b  # b транслюється по рядках
            if i < len(self.weights) - 1:
                # приховані шари — tanh
                a = self.activation(z)
            else:
                # вихідний шар — лінійний
                a = z
        return a


class GeneticAlgorithm:
    def __init__(
        self,
        layer_sizes,
        pop_size=80,
        crossover_rate=0.8,
        mutation_rate=0.05,
        mutation_scale=0.1,
        elitism=4,
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)
        self.nn = FeedForwardNN(layer_sizes)
        self.chrom_length = self.nn.n_params
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism = elitism
        self.population = None

    def _init_population(self):
        # Дійсне кодування: ваги ~ U[-1, 1]
        self.population = self.rng.uniform(-1.0, 1.0, size=(self.pop_size, self.chrom_length))

    def _evaluate_population(self, X, y_true):
        """
        Повертає масив MSE-помилок для кожної хромосоми.
        """
        errors = np.zeros(self.pop_size)
        for i, chrom in enumerate(self.population):
            self.nn.decode_chromosome(chrom)
            y_pred = self.nn.forward(X).reshape(-1)
            errors[i] = np.mean((y_true - y_pred) ** 2)
        return errors

    def _select_parent(self, errors, tournament_size=3):
        """
        Турнірна селекція: обираємо кращу (з меншою помилкою) з k випадкових.
        """
        idx = self.rng.integers(0, self.pop_size, size=tournament_size)
        best_idx = idx[np.argmin(errors[idx])]
        return self.population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        if self.rng.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = self.rng.integers(1, self.chrom_length)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, chromosome):
        """
        Точкова мутація: до частини генів додаємо гаусів шум.
        """
        mask = self.rng.random(self.chrom_length) < self.mutation_rate
        if np.any(mask):
            chromosome[mask] += self.rng.normal(0.0, self.mutation_scale, size=np.sum(mask))
        # обмежимо діапазон ваг для стабільності
        np.clip(chromosome, -5.0, 5.0, out=chromosome)
        return chromosome

    def run(self, X, y_true, generations=200):
        """
        Запускає GA на заданій вибірці (X, y_true).
        Повертає:
          - найкращу знайдену хромосому
          - історію найкращої помилки (MSE) по поколіннях
        """
        self._init_population()
        best_history = []

        for gen in range(generations):
            errors = self._evaluate_population(X, y_true)
            best_idx = np.argmin(errors)
            best_error = errors[best_idx]
            best_history.append(best_error)

            print(f"Покоління {gen:4d}: найкраща MSE = {best_error:.6f}")

            sorted_idx = np.argsort(errors)
            new_population = [self.population[i].copy() for i in sorted_idx[: self.elitism]]

            while len(new_population) < self.pop_size:
                p1 = self._select_parent(errors)
                p2 = self._select_parent(errors)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    c2 = self._mutate(c2)
                    new_population.append(c2)

            self.population = np.vstack(new_population)

        errors = self._evaluate_population(X, y_true)
        best_idx = np.argmin(errors)
        best_chrom = self.population[best_idx].copy()
        final_best_error = errors[best_idx]
        print(f"\nФінальна найкраща MSE = {final_best_error:.6f}")

        self.nn.decode_chromosome(best_chrom)
        return best_chrom, np.array(best_history)


if __name__ == "__main__":
    x_vals = np.linspace(0.0, 5.0, 25)
    y_vals = np.linspace(0.0, 5.0, 25)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    inputs = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    targets = target_function(inputs[:, 0], inputs[:, 1])

    layer_sizes = [2, 4, 8, 10, 6, 6, 1]

    ga = GeneticAlgorithm(
        layer_sizes=layer_sizes,
        pop_size=80,
        crossover_rate=0.8,
        mutation_rate=0.05,
        mutation_scale=0.1,
        elitism=4,
        seed=42,
    )

    best_chrom, history = ga.run(inputs, targets, generations=150)

    rng = np.random.default_rng(0)
    n_test = 10
    x_test = rng.uniform(0.0, 5.0, size=n_test)
    y_test = rng.uniform(0.0, 5.0, size=n_test)
    test_inputs = np.column_stack([x_test, y_test])
    true_vals = target_function(x_test, y_test)
    pred_vals = ga.nn.forward(test_inputs).reshape(-1)

    print("\nДекілька тестових точок (x, y, z_істинне, z_мережа):")
    for i in range(n_test):
        print(
            f"{i:2d}: x={x_test[i]:6.3f}, y={y_test[i]:6.3f}, "
            f"z_true={true_vals[i]:7.4f}, z_pred={pred_vals[i]:7.4f}"
        )
