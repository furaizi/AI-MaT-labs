#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def _mix_param(
    weights: np.ndarray,
    segment_params: Dict[str, Dict],
    seg_names: List[str],
    key: str,
    is_prob: bool = False
) -> float:
    vals = np.array([segment_params[name][key] for name in seg_names], dtype=float)
    v = float((weights * vals).sum())
    if is_prob:
        v = float(np.clip(v, 0.01, 0.99))
    return v


def _mix_category_pref(
    weights: np.ndarray,
    cat_proto: Dict[str, np.ndarray],
    seg_names: List[str]
) -> np.ndarray:
    mats = np.stack([cat_proto[name] for name in seg_names], axis=0)  # S x C
    v = (weights[:, None] * mats).sum(axis=0)
    v = np.clip(v, 1e-9, None)
    return v / v.sum()


def generate_synthetic_purchases(
    n_customers: int = 450,
    n_categories: int = 8,
    months: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Returns a purchases DataFrame, one row per basket.
    Columns: customer_id, day, hour, is_weekend, basket_value, items, promo_items,
             returned_items, cat_0..cat_{n_categories-1}
    """
    rng = np.random.default_rng(seed)

    latent_segments = {
        "discount_hunters": dict(avg_basket_mu=18, avg_basket_sigma=0.35, promo_rate=0.55, returns=0.06, freq=2.0, evening=0.35, weekend=0.50),
        "evening_quick":    dict(avg_basket_mu=22, avg_basket_sigma=0.28, promo_rate=0.20, returns=0.02, freq=2.6, evening=0.70, weekend=0.45),
        "family_weekend":   dict(avg_basket_mu=35, avg_basket_sigma=0.30, promo_rate=0.25, returns=0.03, freq=1.6, evening=0.40, weekend=0.72),
        "returns_prone":    dict(avg_basket_mu=26, avg_basket_sigma=0.40, promo_rate=0.35, returns=0.12, freq=2.2, evening=0.45, weekend=0.48),
    }
    segment_names = list(latent_segments.keys())

    cat_proto = {
        "discount_hunters": rng.dirichlet([2, 4, 3, 2, 3, 2, 2, 2][:n_categories]),
        "evening_quick":    rng.dirichlet([3, 2, 2, 2, 4, 3, 2, 2][:n_categories]),
        "family_weekend":   rng.dirichlet([4, 3, 2, 4, 3, 2, 2, 2][:n_categories]),
        "returns_prone":    rng.dirichlet([2, 2, 4, 2, 2, 3, 3, 2][:n_categories]),
    }

    seg_weights = rng.dirichlet(np.ones(len(segment_names)), size=n_customers)  # (n_customers, S)

    purchase_rows: List[Dict] = []
    for cid in range(n_customers):
        w = seg_weights[cid]
        avg_basket_mu = _mix_param(w, latent_segments, segment_names, "avg_basket_mu")
        avg_basket_sigma = _mix_param(w, latent_segments, segment_names, "avg_basket_sigma")
        promo_rate = _mix_param(w, latent_segments, segment_names, "promo_rate", is_prob=True)
        return_rate = _mix_param(w, latent_segments, segment_names, "returns", is_prob=True)
        freq = _mix_param(w, latent_segments, segment_names, "freq")
        evening_share = _mix_param(w, latent_segments, segment_names, "evening", is_prob=True)
        weekend_share = _mix_param(w, latent_segments, segment_names, "weekend", is_prob=True)
        cat_pref = _mix_category_pref(w, cat_proto, segment_names)

        n_purchases = int(rng.poisson(freq * months) + 1)
        days = rng.integers(0, 30 * months, size=n_purchases)
        dow = days % 7
        is_weekend = (dow >= 5).astype(int)
        weekend_mask = (rng.random(n_purchases) < weekend_share)
        is_weekend = np.where(weekend_mask, 1, is_weekend)

        if evening_share >= 0.5:
            hours = (rng.normal(20, 2.5, size=n_purchases)).astype(int)
        else:
            hours = (rng.normal(13, 3.0, size=n_purchases)).astype(int)
        hours = np.clip(hours, 8, 22)

        basket_vals = rng.lognormal(mean=np.log(avg_basket_mu), sigma=avg_basket_sigma, size=n_purchases)
        items_per_basket = rng.integers(1, 12, size=n_purchases)

        for i in range(n_purchases):
            k = int(items_per_basket[i])
            cats = rng.choice(np.arange(n_categories), size=k, p=cat_pref)
            promo_flags = (rng.random(k) < promo_rate)
            return_flags = (rng.random(k) < return_rate)

            row = {
                "customer_id": cid,
                "day": int(days[i]),
                "hour": int(hours[i]),
                "is_weekend": int(is_weekend[i]),
                "basket_value": float(basket_vals[i]),
                "items": k,
                "promo_items": int(promo_flags.sum()),
                "returned_items": int(return_flags.sum()),
            }
            for c in range(n_categories):
                row[f"cat_{c}"] = int((cats == c).sum())

            purchase_rows.append(row)

    return pd.DataFrame(purchase_rows)


def gini_dispersion(counts: np.ndarray) -> float:
    """Gini-like dispersion: 1 - sum(p_i^2), where p_i are normalized counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def build_customer_features(purchases: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate per-customer features:
    - avg_basket_value
    - category_dispersion (Gini)
    - promo_share
    - mean_interpurchase_days
    - returns_rate
    - evening_share (18..23)
    - weekend_share
    """
    feature_rows: List[Dict] = []
    cat_cols = [c for c in purchases.columns if c.startswith("cat_")]

    for cid, grp in purchases.groupby("customer_id"):
        avg_basket_value = float(grp["basket_value"].mean())
        total_items = int(grp["items"].sum())
        promo_share = float(grp["promo_items"].sum() / max(total_items, 1))
        returns_rate = float(grp["returned_items"].sum() / max(total_items, 1))
        evening_share = float(grp["hour"].between(18, 23).mean())
        weekend_share = float(grp["is_weekend"].mean())

        d = np.sort(grp["day"].values.astype(int))
        if len(d) > 1:
            inter = np.diff(d)
            mean_interpurchase_days = float(np.mean(inter))
        else:
            mean_interpurchase_days = 30.0

        cat_sum = grp[cat_cols].sum().values.astype(float)
        cat_disp = gini_dispersion(cat_sum)

        feature_rows.append({
            "customer_id": cid,
            "avg_basket_value": avg_basket_value,
            "category_dispersion": cat_disp,
            "promo_share": promo_share,
            "mean_interpurchase_days": mean_interpurchase_days,
            "returns_rate": returns_rate,
            "evening_share": evening_share,
            "weekend_share": weekend_share,
            "n_purchases": int(len(grp)),
        })

    feat_df = pd.DataFrame(feature_rows)
    feature_cols = [
        "avg_basket_value",
        "category_dispersion",
        "promo_share",
        "mean_interpurchase_days",
        "returns_rate",
        "evening_share",
        "weekend_share",
    ]
    return feat_df, feature_cols


def fuzzy_c_means(
    X_std: np.ndarray,
    c: int = 4,
    m: float = 2.0,
    maxiter: int = 200,
    error: float = 1e-5,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Simple, stable FCM:
    - Random U initialization (column-stochastic)
    - Iterative updates of centers V, distances D, memberships U
    - Zero-distance handling (one-hot mass on tied minima)
    """
    rng = np.random.default_rng(seed)
    n, d = X_std.shape
    U = rng.random((c, n))
    U /= U.sum(axis=0, keepdims=True)

    J_history: List[float] = []
    power = 1.0 / (m - 1.0)
    eps = 1e-12

    U_prev = None
    for _ in range(maxiter):
        U_m = U ** m
        V = (U_m @ X_std) / np.maximum(U_m.sum(axis=1, keepdims=True), eps)

        diff = X_std[None, :, :] - V[:, None, :]
        D = np.sum(diff * diff, axis=2)

        J = float(np.sum(U_m * D))
        J_history.append(J)

        U_new = np.empty_like(U)
        for k in range(n):
            d_col = D[:, k]
            if np.any(d_col <= eps):
                zeros = (d_col <= eps)
                cnt = int(np.sum(zeros))
                U_new[:, k] = 0.0
                U_new[zeros, k] = 1.0 / cnt
            else:
                ratios = (d_col[:, None] / d_col[None, :]) ** power  # (c, c)
                denom = np.sum(ratios, axis=1)                        # (c,)
                U_new[:, k] = 1.0 / np.maximum(denom, eps)

        if U_prev is not None and np.linalg.norm(U_new - U_prev) < error:
            U = U_new
            break

        U_prev = U_new
        U = U_new

    return V, U, J_history


def fuzzy_partition_coefficient(U: np.ndarray) -> float:
    n = U.shape[1]
    return float(np.sum(U ** 2) / n)


def partition_entropy(U: np.ndarray) -> float:
    n = U.shape[1]
    eps = 1e-12
    return float(-np.sum(U * np.log(U + eps)) / n)


def interpret_centers(
    centers_orig: np.ndarray,
    feature_cols: List[str],
    feat_df: pd.DataFrame
) -> List[str]:
    """Heuristic tags for each center based on deviations from global means."""
    means = feat_df[feature_cols].mean()
    labels: List[str] = []
    for row in centers_orig:
        r = pd.Series(row, index=feature_cols)
        tags = []
        if r["promo_share"] > means["promo_share"] * 1.2:
            tags.append("цінові-мисливці")
        if r["avg_basket_value"] > means["avg_basket_value"] * 1.15:
            tags.append("високий чек")
        elif r["avg_basket_value"] < means["avg_basket_value"] * 0.85:
            tags.append("низький чек")
        if r["evening_share"] > 0.55:
            tags.append("вечірні покупки")
        if r["weekend_share"] > 0.60:
            tags.append("вихідні")
        if r["mean_interpurchase_days"] < means["mean_interpurchase_days"] * 0.8:
            tags.append("часті покупки")
        if r["returns_rate"] > means["returns_rate"] * 1.5:
            tags.append("схильні до повернень")
        if r["category_dispersion"] > means["category_dispersion"] * 1.1:
            tags.append("різноманітні категорії")

        labels.append(", ".join(tags) if tags else "змішаний профіль")
    return labels


def main():
    parser = argparse.ArgumentParser(description="FCM clustering for POS basket behavior.")
    parser.add_argument("--customers", type=int, default=450, help="Number of customers (synthetic)")
    parser.add_argument("--categories", type=int, default=8, help="Number of product categories")
    parser.add_argument("--months", type=int, default=3, help="Simulation span in months")
    parser.add_argument("--clusters", type=int, default=4, choices=[3, 4, 5], help="Number of clusters (K)")
    parser.add_argument("--m", type=float, default=2.0, help="Fuzzifier m (>1)")
    parser.add_argument("--maxiter", type=int, default=300, help="Max FCM iterations")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance on ΔU")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output", type=str, default="../output", help="Output folder for plots and CSVs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    purchases = generate_synthetic_purchases(
        n_customers=args.customers,
        n_categories=args.categories,
        months=args.months,
        seed=args.seed
    )

    feat_df, feature_cols = build_customer_features(purchases)
    X = feat_df[feature_cols].values

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    centers_std, U, J_hist = fuzzy_c_means(
        X_std,
        c=args.clusters,
        m=args.m,
        maxiter=args.maxiter,
        error=args.tol,
        seed=args.seed
    )
    centers_orig = scaler.inverse_transform(centers_std)

    fpc = fuzzy_partition_coefficient(U)
    pe = partition_entropy(U)

    # Objective curve
    plt.figure()
    plt.plot(J_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Objective J_m")
    plt.title("FCM convergence (J_m vs iteration)")
    plt.grid(True)
    fig1_path = os.path.join(args.output, "fcm_objective.png")
    plt.savefig(fig1_path, bbox_inches="tight")
    plt.close()

    # PCA scatter with argmax memberships
    labels = np.argmax(U, axis=0)
    pca = PCA(n_components=2, random_state=args.seed)
    X_2d = pca.fit_transform(X_std)
    centers_2d = pca.transform(centers_std)

    plt.figure()
    for k in range(args.clusters):
        idx = np.where(labels == k)[0]
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=12, label=f"Cluster {k}")
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker="*", s=180, label="Centers")
    plt.legend()
    plt.title("PCA projection (customers & centers, argmax μ)")
    fig2_path = os.path.join(args.output, "fcm_pca.png")
    plt.savefig(fig2_path, bbox_inches="tight")
    plt.close()

    centers_df = pd.DataFrame(centers_orig, columns=feature_cols)
    centers_df.insert(0, "cluster", range(args.clusters))
    centers_df["interpretation"] = interpret_centers(centers_orig, feature_cols, feat_df)
    centers_df.to_csv(os.path.join(args.output, "centers.csv"), index=False)

    feat_df.to_csv(os.path.join(args.output, "features.csv"), index=False)

    memberships_df = pd.DataFrame({"customer_id": feat_df["customer_id"]})
    for j in range(args.clusters):
        memberships_df[f"mu_{j}"] = U[j]
    memberships_df["label"] = labels
    memberships_df.to_csv(os.path.join(args.output, "memberships.csv"), index=False)

    print("=== FCM done ===")
    print(f"Clusters (K)        : {args.clusters}")
    print(f"m (fuzzifier)       : {args.m}")
    print(f"Iterations          : {len(J_hist)}")
    print(f"FPC (higher better) : {fpc:.4f}")
    print(f"PE  (lower better)  : {pe:.4f}")
    print(f"Objective plot      : {fig1_path}")
    print(f"PCA scatter         : {fig2_path}")
    print(f"Centers CSV         : {os.path.join(args.output, 'centers.csv')}")
    print(f"Features CSV        : {os.path.join(args.output, 'features.csv')}")
    print(f"Memberships CSV     : {os.path.join(args.output, 'memberships.csv')}")


if __name__ == "__main__":
    main()
