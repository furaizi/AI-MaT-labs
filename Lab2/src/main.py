import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Tuple
from pathlib import Path

@dataclass
class TriangularMF:
    name: str
    a: float
    b: float
    c: float
    def mu(self, x: np.ndarray) -> np.ndarray:
        left = (x - self.a) / (self.b - self.a) if self.b != self.a else np.where(x <= self.b, 1.0, 0.0)
        right = (self.c - x) / (self.c - self.b) if self.c != self.b else np.where(x >= self.b, 1.0, 0.0)
        return np.maximum(0.0, np.minimum(left, right)).astype(float)

@dataclass
class TrapezoidalMF:
    name: str
    a: float
    b: float
    c: float
    d: float
    def mu(self, x: np.ndarray) -> np.ndarray:
        rise = (x - self.a) / (self.b - self.a) if self.b != self.a else np.where(x >= self.b, 1.0, 0.0)
        fall = (self.d - x) / (self.d - self.c) if self.d != self.c else np.where(x <= self.c, 1.0, 0.0)
        return np.maximum(0.0, np.minimum(np.minimum(rise, 1.0), np.minimum(fall, 1.0))).astype(float)

@dataclass
class GaussianMF:
    name: str
    c: float
    sigma: float
    def mu(self, x: np.ndarray) -> np.ndarray:
        s = self.sigma if self.sigma > 1e-12 else 1e-12
        return np.exp(-0.5 * ((x - self.c) / s) ** 2).astype(float)

def make_triangular_mfs(prefix: str, xmin: float, xmax: float, n: int) -> Tuple[List[TriangularMF], np.ndarray]:
    centers = np.linspace(xmin, xmax, n)
    mids = (centers[:-1] + centers[1:]) / 2.0
    mfs = []
    for i, c in enumerate(centers):
        a = xmin if i == 0 else mids[i-1]
        d = xmax if i == n-1 else mids[i]
        mfs.append(TriangularMF(f"{prefix}{i+1}", a, c, d))
    return mfs, centers

def make_trapezoidal_mfs(prefix: str, xmin: float, xmax: float, n: int) -> Tuple[List[TrapezoidalMF], np.ndarray]:
    centers = np.linspace(xmin, xmax, n)
    mids = (centers[:-1] + centers[1:]) / 2.0
    mfs = []
    for i, c in enumerate(centers):
        a = xmin if i == 0 else mids[i-1]
        d = xmax if i == n-1 else mids[i]
        width = d - a
        b = a + 0.3 * width
        cc = d - 0.3 * width
        b = max(a, min(b, cc))
        cc = min(d, max(cc, b))
        mfs.append(TrapezoidalMF(f"{prefix}{i+1}", a, b, cc, d))
    return mfs, centers

def make_gaussian_mfs(prefix: str, xmin: float, xmax: float, n: int) -> Tuple[List[GaussianMF], np.ndarray]:
    centers = np.linspace(xmin, xmax, n)
    spacing = np.mean(np.diff(centers)) if n > 1 else (xmax - xmin) / 2.0
    sigma = spacing / np.sqrt(2.0 * np.log(2.0))
    mfs = [GaussianMF(f"{prefix}{i+1}", c, sigma) for i, c in enumerate(centers)]
    return mfs, centers


class MamdaniFuzzySystem:
    def __init__(self,
                 x_mfs,
                 y_mfs,
                 f_mfs,
                 f_rule_selector: Callable[[float, float], float],
                 f_grid: np.ndarray):
        self.x_mfs = x_mfs
        self.y_mfs = y_mfs
        self.f_mfs = f_mfs
        self.f_grid = f_grid
        self.rule_table_idx, self.rule_table_names = self._build_rule_table(f_rule_selector)

    def _build_rule_table(self, f_rule_selector):
        def peak(mf):
            if isinstance(mf, TriangularMF): return mf.b
            if isinstance(mf, GaussianMF):  return mf.c
            if isinstance(mf, TrapezoidalMF): return 0.5 * (mf.b + mf.c)
            raise ValueError("Unknown MF")
        x_peaks = [peak(mf) for mf in self.x_mfs]
        y_peaks = [peak(mf) for mf in self.y_mfs]

        def choose_out_idx(val: float) -> int:
            vals = np.array([mf.mu(np.array([val]))[0] for mf in self.f_mfs])
            return int(np.argmax(vals))

        idx = np.zeros((len(self.y_mfs), len(self.x_mfs)), dtype=int)
        name_rows = []
        for j, yv in enumerate(y_peaks):
            row = []
            for i, xv in enumerate(x_peaks):
                f_val = f_rule_selector(xv, yv)
                k = choose_out_idx(f_val)
                idx[j, i] = k
                row.append(self.f_mfs[k].name)
            name_rows.append(row)
        df = pd.DataFrame(name_rows,
                          index=[mf.name for mf in self.y_mfs],
                          columns=[mf.name for mf in self.x_mfs])
        return idx, df

    def infer(self, x_val: float, y_val: float, diagonal_only: bool=False) -> float:
        mu_x = np.array([mf.mu(np.array([x_val]))[0] for mf in self.x_mfs])
        mu_y = np.array([mf.mu(np.array([y_val]))[0] for mf in self.y_mfs])
        agg = np.zeros_like(self.f_grid, dtype=float)
        for j in range(len(self.y_mfs)):
            for i in range(len(self.x_mfs)):
                if diagonal_only and i != j:
                    continue
                w = min(mu_x[i], mu_y[j])
                if w <= 0:
                    continue
                k = self.rule_table_idx[j, i]
                mu_out = self.f_mfs[k].mu(self.f_grid)
                agg = np.maximum(agg, np.minimum(w, mu_out))
        s = agg.sum()
        return 0.0 if s <= 1e-12 else float((self.f_grid * agg).sum() / s)


def main(output_dir: Path = Path("output")):
    # Prepare output folders
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Universes
    x_min, x_max = 0.0, 20.0
    y_min, y_max = -1.0, 1.0
    f_min, f_max = -1.0, 1.0

    # Discretization
    x_eval = np.linspace(x_min, x_max, 1001)
    y_true = np.sin(np.abs(x_eval)) * np.cos(x_eval / 2.0)
    z_true = y_true * np.sin(x_eval)

    # Output grid for defuzzification
    f_grid = np.linspace(f_min, f_max, 601)

    # Target function for rules
    def f_xy(x, y):
        return y * np.sin(x)

    # Build 3 systems with different MF shapes
    x_tri, _ = make_triangular_mfs("mx", x_min, x_max, 6)
    y_tri, _ = make_triangular_mfs("my", y_min, y_max, 6)
    f_tri, _ = make_triangular_mfs("mf", f_min, f_max, 9)
    sys_tri = MamdaniFuzzySystem(x_tri, y_tri, f_tri, f_xy, f_grid)

    x_trap, _ = make_trapezoidal_mfs("mx", x_min, x_max, 6)
    y_trap, _ = make_trapezoidal_mfs("my", y_min, y_max, 6)
    f_trap, _ = make_trapezoidal_mfs("mf", f_min, f_max, 9)
    sys_trap = MamdaniFuzzySystem(x_trap, y_trap, f_trap, f_xy, f_grid)

    x_gaus, _ = make_gaussian_mfs("mx", x_min, x_max, 6)
    y_gaus, _ = make_gaussian_mfs("my", y_min, y_max, 6)
    f_gaus, _ = make_gaussian_mfs("mf", f_min, f_max, 9)
    sys_gaus = MamdaniFuzzySystem(x_gaus, y_gaus, f_gaus, f_xy, f_grid)

    systems = {
        "triangular": sys_tri,
        "trapezoidal": sys_trap,
        "gaussian": sys_gaus
    }

    # System evaluation
    def eval_system(sys: MamdaniFuzzySystem, diagonal_only: bool=False):
        z_pred = np.array([sys.infer(float(x), float(y_true[i]), diagonal_only=diagonal_only)
                           for i, x in enumerate(x_eval)])
        denom = np.abs(z_true + 1.0)
        denom = np.where(denom < 1e-6, 1e-6, denom)
        rel = np.abs((z_true + 1.0) - (z_pred + 1.0)) / denom * 100.0
        return z_pred, rel

    results = {}
    for name, sys in systems.items():
        z_pred, rel = eval_system(sys, diagonal_only=False)
        results[name] = {"z_pred": z_pred, "rel": rel,
                         "mean": float(rel.mean()), "median": float(np.median(rel))}

    # Reduced rule base: only diagonal (for triangular MFs)
    z_pred_d, rel_d = eval_system(sys_tri, diagonal_only=True)
    results["triangular_diagonal_only"] = {"z_pred": z_pred_d, "rel": rel_d,
                                           "mean": float(rel_d.mean()), "median": float(np.median(rel_d))}

    # Print summary
    for k, v in results.items():
        print(f"{k}: mean ε%={v['mean']:.2f}, median ε%={v['median']:.2f}")

    # Save artifacts
    # True curves
    plt.figure(); plt.plot(x_eval, y_true, label="y(x)"); plt.title("y(x) = sin(|x|)·cos(x/2)"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.savefig(figures_dir / "y_true.png", dpi=180); plt.close()
    plt.figure(); plt.plot(x_eval, z_true, label="z_true"); plt.title("z(x) = y(x)·sin(x)"); plt.xlabel("x"); plt.ylabel("z"); plt.legend(); plt.savefig(figures_dir / "z_true.png", dpi=180); plt.close()

    # Prediction comparison
    for name in ["triangular", "trapezoidal", "gaussian"]:
        plt.figure(); plt.plot(x_eval, z_true, label="z_true"); plt.plot(x_eval, results[name]["z_pred"], label=f"z_pred ({name})")
        plt.title(f"True vs Predicted — {name}"); plt.xlabel("x"); plt.ylabel("z"); plt.legend(); plt.savefig(figures_dir / f"z_true_vs_pred_{name}.png", dpi=180); plt.close()

    # Full vs diagonal rule base (triangular)
    plt.figure(); plt.plot(x_eval, z_true, label="z_true")
    plt.plot(x_eval, results["triangular"]["z_pred"], label="triangular (full)")
    plt.plot(x_eval, results["triangular_diagonal_only"]["z_pred"], label="triangular (diagonal-only)")
    plt.title("Triangular: full vs diagonal-only"); plt.xlabel("x"); plt.ylabel("z"); plt.legend(); plt.savefig(figures_dir / "tri_full_vs_diag.png", dpi=180); plt.close()

    # Error curves
    plt.figure()
    for name, v in results.items():
        plt.plot(x_eval, v["rel"], label=name)
    plt.title("Relative error (%) vs x"); plt.xlabel("x"); plt.ylabel("ε, %"); plt.legend(); plt.savefig(figures_dir / "errors.png", dpi=180); plt.close()

    # Tables and rules (to output/)
    (output_dir / "rule_table_triangular.csv").write_text(
        sys_tri.rule_table_names.to_csv(index=True), encoding="utf-8"
    )
    pd.DataFrame([{"model": k, "mean_rel_err_%": v["mean"], "median_rel_err_%": v["median"]}
                  for k, v in results.items()]).to_csv(output_dir / "error_summary.csv", index=False)
    with (output_dir / "triangular_rules_36.txt").open("w", encoding="utf-8") as f:
        for j, y_mf in enumerate(sys_tri.y_mfs):
            for i, x_mf in enumerate(sys_tri.x_mfs):
                k = sys_tri.rule_table_idx[j, i]
                f.write(f"If (x is {sys_tri.x_mfs[i].name}) and (y is {sys_tri.y_mfs[j].name}) then (f is {sys_tri.f_mfs[k].name}).\n")

if __name__ == "__main__":
    main()
