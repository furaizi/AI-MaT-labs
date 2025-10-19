import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
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
    mfs: List[TriangularMF] = []
    for i, c in enumerate(centers):
        a = xmin if i == 0 else mids[i - 1]
        d = xmax if i == n - 1 else mids[i]
        mfs.append(TriangularMF(f"{prefix}{i + 1}", a, c, d))
    return mfs, centers


def make_trapezoidal_mfs(prefix: str, xmin: float, xmax: float, n: int) -> Tuple[List[TrapezoidalMF], np.ndarray]:
    centers = np.linspace(xmin, xmax, n)
    mids = (centers[:-1] + centers[1:]) / 2.0
    mfs: List[TrapezoidalMF] = []
    for i, _ in enumerate(centers):
        a = xmin if i == 0 else mids[i - 1]
        d = xmax if i == n - 1 else mids[i]
        w = d - a
        b = a + 0.3 * w
        c = d - 0.3 * w
        b = max(a, min(b, c))
        c = min(d, max(c, b))
        mfs.append(TrapezoidalMF(f"{prefix}{i + 1}", a, b, c, d))
    return mfs, centers


def make_gaussian_mfs(prefix: str, xmin: float, xmax: float, n: int) -> Tuple[List[GaussianMF], np.ndarray]:
    centers = np.linspace(xmin, xmax, n)
    spacing = np.mean(np.diff(centers)) if n > 1 else (xmax - xmin) / 2.0
    sigma = spacing / np.sqrt(2.0 * np.log(2.0))
    mfs = [GaussianMF(f"{prefix}{i + 1}", c, sigma) for i, c in enumerate(centers)]
    return mfs, centers


class MamdaniFuzzySystem:
    def __init__(
        self,
        x_mfs,
        y_mfs,
        z_mfs,
        z_rule_selector: Callable[[float, float], float],
        z_grid: np.ndarray,
    ):
        self.x_mfs = x_mfs
        self.y_mfs = y_mfs
        self.z_mfs = z_mfs
        self.z_grid = z_grid
        self.rule_table_idx, self.rule_table_names = self._build_rule_table(z_rule_selector)

    def _peak(self, mf):
        if isinstance(mf, TriangularMF):
            return mf.b
        if isinstance(mf, GaussianMF):
            return mf.c
        if isinstance(mf, TrapezoidalMF):
            return 0.5 * (mf.b + mf.c)
        raise ValueError("Unknown MF")

    def _build_rule_table(self, z_rule_selector):
        x_peaks = [self._peak(mf) for mf in self.x_mfs]
        y_peaks = [self._peak(mf) for mf in self.y_mfs]

        def choose_out_idx(val: float) -> int:
            vals = np.array([mf.mu(np.array([val]))[0] for mf in self.z_mfs])
            return int(np.argmax(vals))

        idx = np.zeros((len(self.y_mfs), len(self.x_mfs)), dtype=int)
        name_rows: List[List[str]] = []
        for j, yv in enumerate(y_peaks):
            row = []
            for i, xv in enumerate(x_peaks):
                z_val = z_rule_selector(xv, yv)
                k = choose_out_idx(z_val)
                idx[j, i] = k
                row.append(self.z_mfs[k].name)
            name_rows.append(row)

        df = pd.DataFrame(
            name_rows, index=[mf.name for mf in self.y_mfs], columns=[mf.name for mf in self.x_mfs]
        )
        return idx, df

    def infer(self, x_val: float, y_val: float, diagonal_only: bool = False) -> float:
        mu_x = np.array([mf.mu(np.array([x_val]))[0] for mf in self.x_mfs])
        mu_y = np.array([mf.mu(np.array([y_val]))[0] for mf in self.y_mfs])
        agg = np.zeros_like(self.z_grid, dtype=float)
        for j in range(len(self.y_mfs)):
            for i in range(len(self.x_mfs)):
                if diagonal_only and i != j:
                    continue
                w = min(mu_x[i], mu_y[j])
                if w <= 0:
                    continue
                k = self.rule_table_idx[j, i]
                mu_out = self.z_mfs[k].mu(self.z_grid)
                agg = np.maximum(agg, np.minimum(w, mu_out))
        s = agg.sum()
        return 0.0 if s <= 1e-12 else float((self.z_grid * agg).sum() / s)


def plot_mfs(mfs, domain: Tuple[float, float], title: str, save_path: Path) -> None:
    xs = np.linspace(domain[0], domain[1], 1000)
    plt.figure()
    for mf in mfs:
        plt.plot(xs, mf.mu(xs), label=mf.name)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Membership")
    plt.legend(loc="best")
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def _peak(mf):
    if isinstance(mf, TriangularMF):
        return mf.b
    if isinstance(mf, GaussianMF):
        return mf.c
    if isinstance(mf, TrapezoidalMF):
        return 0.5 * (mf.b + mf.c)
    raise ValueError("Unknown MF")


def build_values_table(x_mfs, y_mfs, func: Callable[[float, float], float]) -> pd.DataFrame:
    x_peaks = [_peak(mf) for mf in x_mfs]
    y_peaks = [_peak(mf) for mf in y_mfs]
    rows = [[func(xv, yv) for xv in x_peaks] for yv in y_peaks]
    df = pd.DataFrame(rows, index=[f"{yv:.2f}" for yv in y_peaks], columns=[f"{xv:.0f}" for xv in x_peaks])
    df.index.name = "y\\x"
    return df


def render_ascii_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    data = [[idx] + [df.loc[idx, c] for c in df.columns] for idx in df.index]
    sdata = []
    for r in data:
        row = []
        for cell in r:
            row.append(f"{cell:.2f}" if isinstance(cell, (float, np.floating)) else str(cell))
        sdata.append(row)
    col_widths = [
        max(len(str(cols[i])) if i < len(cols) else 0, max(len(r[i]) for r in sdata)) for i in range(len(cols))
    ]

    def hline():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def rowfmt(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)) + " |"

    lines = [hline(), rowfmt(cols), hline()]
    lines += [rowfmt(r) for r in sdata]
    lines.append(hline())
    return "\n".join(lines)


def dump_tables_and_rules(
    shape: str,
    sys: MamdaniFuzzySystem,
    x_mfs,
    y_mfs,
    rule_func: Callable[[float, float], float],
    base_out: Path,
) -> None:
    shape_dir = base_out / shape
    figs_dir = shape_dir / "figures"
    shape_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    values_df = build_values_table(x_mfs, y_mfs, rule_func)
    ascii_vals = render_ascii_table(values_df)
    (shape_dir / f"values_table_{shape}.txt").write_text(ascii_vals, encoding="utf-8")

    names_df = sys.rule_table_names.copy()
    names_df.index.name = "y\\x"
    ascii_names = render_ascii_table(names_df.astype(str))
    (shape_dir / f"names_table_{shape}.txt").write_text(ascii_names, encoding="utf-8")

    rules = []
    for j in range(len(sys.y_mfs)):
        for i in range(len(sys.x_mfs)):
            k = sys.rule_table_idx[j, i]
            rules.append(f"if (x is {sys.x_mfs[i].name}) and (y is {sys.y_mfs[j].name}) then (z is {sys.z_mfs[k].name})")
    (shape_dir / f"{shape}_rules_36.txt").write_text("\n".join(rules), encoding="utf-8")

    plot_mfs(x_mfs, (0.0, 20.0), f"{shape} – x MFs", figs_dir / f"mfs_x_{shape}.png")
    plot_mfs(y_mfs, (-1.0, 1.0), f"{shape} – y MFs", figs_dir / f"mfs_y_{shape}.png")
    plot_mfs(sys.z_mfs, (-1.0, 1.0), f"{shape} – z MFs", figs_dir / f"mfs_z_{shape}.png")


def evaluate_and_save_plots(
    x_eval: np.ndarray,
    y_true: np.ndarray,
    z_true: np.ndarray,
    results: Dict[str, Dict],
    base_out: Path,
) -> None:
    common_dir = base_out / "common"
    common_dir.mkdir(parents=True, exist_ok=True)

    # save y_true plot
    plt.figure()
    plt.plot(x_eval, y_true, label="y(x)")
    plt.title("y(x) = sin(|x|) * cos(x/2)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(common_dir / "y_true.png", dpi=180)
    plt.close()

    # save z_true plot
    plt.figure()
    plt.plot(x_eval, z_true, label="z_true")
    plt.title("z(x) = y(x) * sin(x)")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    plt.savefig(common_dir / "z_true.png", dpi=180)
    plt.close()

    # global errors plot (comparative)
    plt.figure()
    for name, v in results.items():
        plt.plot(x_eval, v["rel"], label=name)
    plt.title("Relative error (%) vs x")
    plt.xlabel("x")
    plt.ylabel("ε, %")
    plt.legend()
    plt.savefig(common_dir / "errors.png", dpi=180)
    plt.close()

    lines = ["Model\tmean_rel_err_%\tmedian_rel_err_%"]
    for k, v in results.items():
        lines.append(f"{k}\t{v['mean']:.4f}\t{v['median']:.4f}")
    (common_dir / "errors.txt").write_text("\n".join(lines), encoding="utf-8")

    for name, v in results.items():
        if name.endswith("_diagonal_only"):
            continue
        shape_dir = base_out / name
        figs_dir = shape_dir / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(x_eval, z_true, label="z_true")
        plt.plot(x_eval, v["z_pred"], label=f"z_pred ({name})")
        plt.title(f"True vs Predicted — {name}")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.savefig(figs_dir / f"z_true_vs_pred_{name}.png", dpi=180)
        plt.close()

    if "triangular" in results and "triangular_diagonal_only" in results:
        tri_dir = base_out / "triangular" / "figures"
        tri_dir.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(x_eval, z_true, label="z_true")
        plt.plot(x_eval, results["triangular"]["z_pred"], label="triangular (full)")
        plt.plot(x_eval, results["triangular_diagonal_only"]["z_pred"], label="triangular (diagonal-only)")
        plt.title("Triangular: full vs diagonal-only")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.savefig(tri_dir / "tri_full_vs_diag.png", dpi=180)
        plt.close()


def main(output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_min, x_max = 0.0, 20.0
    y_min, y_max = -1.0, 1.0
    z_min, z_max = -1.0, 1.0

    x_eval = np.linspace(x_min, x_max, 1001)
    y_true = np.sin(np.abs(x_eval)) * np.cos(x_eval / 2.0)
    z_true = y_true * np.sin(x_eval)
    z_grid = np.linspace(z_min, z_max, 601)

    def z_xy(x, y):
        return y * np.sin(x)

    makers = {
        "triangular": make_triangular_mfs,
        "trapezoidal": make_trapezoidal_mfs,
        "gaussian": make_gaussian_mfs,
    }

    systems: Dict[str, MamdaniFuzzySystem] = {}
    mf_collections = {}

    for shape, maker in makers.items():
        x_mfs, _ = maker("mx", x_min, x_max, 6)
        y_mfs, _ = maker("my", y_min, y_max, 6)
        z_mfs, _ = maker("mz", z_min, z_max, 9)
        sys = MamdaniFuzzySystem(x_mfs, y_mfs, z_mfs, z_xy, z_grid)
        systems[shape] = sys
        mf_collections[shape] = (x_mfs, y_mfs, z_mfs)

    results: Dict[str, Dict] = {}
    for name, sys in systems.items():
        z_pred = np.array([sys.infer(float(x), float(y_true[i])) for i, x in enumerate(x_eval)])
        denom = np.abs(z_true + 1.0)
        denom = np.where(denom < 1e-6, 1e-6, denom)
        rel = np.abs((z_true + 1.0) - (z_pred + 1.0)) / denom * 100.0
        results[name] = {"z_pred": z_pred, "rel": rel, "mean": float(rel.mean()), "median": float(np.median(rel))}

    z_pred_d = np.array([systems["triangular"].infer(float(x), float(y_true[i]), diagonal_only=True) for i, x in enumerate(x_eval)])
    denom = np.abs(z_true + 1.0)
    denom = np.where(denom < 1e-6, 1e-6, denom)
    rel_d = np.abs((z_true + 1.0) - (z_pred_d + 1.0)) / denom * 100.0
    results["triangular_diagonal_only"] = {"z_pred": z_pred_d, "rel": rel_d, "mean": float(rel_d.mean()), "median": float(np.median(rel_d))}

    evaluate_and_save_plots(x_eval, y_true, z_true, results, output_dir)

    for shape, (x_mfs, y_mfs, z_mfs) in mf_collections.items():
        dump_tables_and_rules(shape, systems[shape], x_mfs, y_mfs, z_xy, output_dir)


if __name__ == "__main__":
    main()
