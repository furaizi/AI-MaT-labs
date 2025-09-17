import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

x = np.linspace(0, 10, 1001)

params = {
    "trimf": [2.0, 5.0, 8.0],
    "trapmf": [1.0, 3.0, 7.0, 9.0],
    "gaussmf": {"mean": 5.0, "sigma": 1.2},
    "gauss2mf": {"mean1": 3.0, "sigma1": 0.8, "mean2": 7.0, "sigma2": 1.0},
    "gbellmf": {"a": 1.5, "b": 2.5, "c": 6.0},
    "sig_right": {"b": 3.0, "c": 5.0},    # відкрита праворуч
    "sig_left": {"b": -3.0, "c": 5.0},    # відкрита ліворуч
    "dsigmf": {"b1": 3.0, "c1": 4.0, "b2": 3.0, "c2": 6.0},
    "psigmf": {"b1": 4.0, "c1": 3.0, "b2": -2.0, "c2": 7.0},
    "zmf": [2.0, 6.0],
    "smf": [4.0, 8.0],
    "pimf": [3.0, 4.0, 6.0, 7.0],
}

y_trim  = fuzz.trimf(x, params["trimf"])
y_trap  = fuzz.trapmf(x, params["trapmf"])
y_gauss = fuzz.gaussmf(x, params["gaussmf"]["mean"], params["gaussmf"]["sigma"])
y_gauss2 = fuzz.gauss2mf(x,
                         params["gauss2mf"]["mean1"], params["gauss2mf"]["sigma1"],
                         params["gauss2mf"]["mean2"], params["gauss2mf"]["sigma2"])
y_gbell = fuzz.gbellmf(x, params["gbellmf"]["a"], params["gbellmf"]["b"], params["gbellmf"]["c"])
y_sig_right = fuzz.sigmf(x, params["sig_right"]["b"], params["sig_right"]["c"])
y_sig_left  = fuzz.sigmf(x, params["sig_left"]["b"],  params["sig_left"]["c"])
y_dsig = fuzz.dsigmf(x, params["dsigmf"]["b1"], params["dsigmf"]["c1"],
                        params["dsigmf"]["b2"], params["dsigmf"]["c2"])
y_psig = fuzz.psigmf(x, params["psigmf"]["b1"], params["psigmf"]["c1"],
                        params["psigmf"]["b2"], params["psigmf"]["c2"])
y_z = fuzz.zmf(x, *params["zmf"])
y_s = fuzz.smf(x, *params["smf"])
y_pi = fuzz.pimf(x, *params["pimf"])

# Дві базові множини для операцій
A = fuzz.gaussmf(x, 4.0, 1.0)
B = fuzz.gaussmf(x, 6.0, 1.5)

# Мінімаксна інтерпретація
and_min = np.minimum(A, B)
or_max  = np.maximum(A, B)

# Імовірнісна інтерпретація
and_prod = A * B
or_prob  = A + B - A * B

not_A = 1.0 - A

OUT = "lab1_figs"
os.makedirs(OUT, exist_ok=True)

def save_plot(x, curves, title, fname):
    plt.figure()
    for y, label, ls in curves:
        plt.plot(x, y, linestyle=ls, label=label)
    plt.ylim(-0.05, 1.05)
    plt.xlim(x.min(), x.max())
    plt.xlabel("x")
    plt.ylabel("μ(x)")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.close()

save_plot(x, [(y_trim, "trimf[2,5,8]", "-")], "Трикутна функція приналежності", "01_trimf.png")
save_plot(x, [(y_trap, "trapmf[1,3,7,9]", "-")], "Трапецієподібна функція приналежності", "02_trapmf.png")

save_plot(x, [(y_gauss, "gaussmf(m=5,σ=1.2)", "-")], "Проста ґаусова ФП", "03_gaussmf.png")
save_plot(x, [(y_gauss2, "gauss2mf(m1=3,σ1=0.8; m2=7,σ2=1.0)", "-")], "Двостороння ґаусова ФП", "04_gauss2mf.png")

save_plot(x, [(y_gbell, "gbellmf(a=1.5,b=2.5,c=6)", "-")], "ФП Узагальнений дзвін", "05_gbellmf.png")

save_plot(x, [(y_sig_right, "sigmf(b=3,c=5)", "-")], "Сигмоїдна (відкрита праворуч)", "06_sigmf_right.png")
save_plot(x, [(y_sig_left,  "sigmf(b=-3,c=5)", "-")], "Сигмоїдна (відкрита ліворуч)", "07_sigmf_left.png")
save_plot(x, [(y_dsig, "dsigmf(b1=3,c1=4,b2=3,c2=6)", "-")], "Двостороння сигмоїдна", "08_dsigmf.png")
save_plot(x, [(y_psig, "psigmf(b1=4,c1=3,b2=-2,c2=7)", "-")], "Несиметрична сигмоїдна", "09_psigmf.png")

save_plot(x, [(y_z, "zmf[2,6]", "-")], "Z-функція", "10_zmf.png")
save_plot(x, [(y_pi, "pimf[3,4,6,7]", "-")], "PI-функція", "11_pimf.png")
save_plot(x, [(y_s, "smf[4,8]", "-")], "S-функція", "12_smf.png")

save_plot(x, [(A, "A (Gauss 4,1)", "--"), (B, "B (Gauss 6,1.5)", "--"), (and_min, "AND = min(A,B)", "-")],
          "Мінімаксна інтерпретація: AND", "13_and_min.png")
save_plot(x, [(A, "A", "--"), (B, "B", "--"), (or_max, "OR = max(A,B)", "-")],
          "Мінімаксна інтерпретація: OR", "14_or_max.png")

save_plot(x, [(A, "A", "--"), (B, "B", "--"), (and_prod, "AND = A·B", "-")],
          "Імовірнісна інтерпретація: AND", "15_and_prod.png")
save_plot(x, [(A, "A", "--"), (B, "B", "--"), (or_prob, "OR = A + B - A·B", "-")],
          "Імовірнісна інтерпретація: OR", "16_or_prob.png")

save_plot(x, [(A, "A", "--"), (1.0 - A, "NOT A = 1 - A", "-")],
          "Доповнення нечіткої множини", "17_not_A.png")

print(f"Готово. Рисунки збережено у папці: {OUT}/")
