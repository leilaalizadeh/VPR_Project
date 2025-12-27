import os
import re
import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# 1) Loading utilities
# ----------------------------

def load_z_data(z_path: str) -> dict:
    z = torch.load(z_path, map_location="cpu", weights_only=False)
    if not isinstance(z, dict):
        raise TypeError(f"Expected dict in {z_path}, got {type(z)}")
    for k in ["predictions", "positives_per_query"]:
        if k not in z:
            raise KeyError(f"Missing key '{k}' in z_data. Found keys: {list(z.keys())}")
    return z


def natural_key(filename: str):
    m = re.search(r"(\d+)", filename)
    return (int(m.group(1)) if m else math.inf, filename)


def load_matches_dir(matches_dir: str, expected_num_queries: int, ext: str = ".torch") -> list:
   
    files = [f for f in os.listdir(matches_dir) if f.endswith(ext)]
    if not files:
        raise FileNotFoundError(f"No *{ext} files found in: {matches_dir}")

    idx_to_file = {}
    for f in files:
        m = re.search(r"(\d+)", f)
        if not m:
            continue
        idx = int(m.group(1))
        idx_to_file[idx] = f

    missing = [i for i in range(expected_num_queries) if i not in idx_to_file]
    if missing:
        raise ValueError(f"Missing match files for query indices: {missing[:20]} (showing up to 20)")

    # load in exact query order 0..expected-1
    all_matches = []
    for i in range(expected_num_queries):
        path = os.path.join(matches_dir, idx_to_file[i])
        obj = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(obj, list) or len(obj) == 0 or "num_inliers" not in obj[0]:
            raise ValueError(f"{path} has unexpected format (need list of dicts with 'num_inliers')")

        all_matches.append(obj)

    extras = sorted([k for k in idx_to_file.keys() if k >= expected_num_queries])
    if extras:
        print(f"[INFO] Ignoring extra match files with indices >= {expected_num_queries}: {extras[:10]}{'...' if len(extras)>10 else ''}", flush=True)

    return all_matches

# ----------------------------
# 2) Core metrics
# ----------------------------

def to_numpy_2d(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    else:
        x = np.array(x)
    if x.ndim != 2:
        raise ValueError(f"predictions must be 2D [num_queries, topK], got shape {x.shape}")
    return x


def correct_at_1(predictions_2d: np.ndarray, positives_per_query: list) -> np.ndarray:
    n = predictions_2d.shape[0]
    if len(positives_per_query) != n:
        raise ValueError(f"positives_per_query length {len(positives_per_query)} != num_queries {n}")

    correct = np.zeros(n, dtype=bool)
    for q in range(n):
        top1 = int(predictions_2d[q, 0])
        pos = positives_per_query[q]
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().tolist()
        correct[q] = top1 in set(pos)
    #retrieval top-1 correct or not
    return correct


def get_inliers_top1(matches_per_query: list) -> np.ndarray:
    n = len(matches_per_query)
    inliers = np.zeros(n, dtype=float)
    for q in range(n):
        inliers[q] = float(matches_per_query[q][0]["num_inliers"])
    return inliers


def reranked_top1_from_inliers(predictions_2d: np.ndarray, matches_per_query: list) -> np.ndarray:
    n, topK = predictions_2d.shape
    reranked = np.zeros(n, dtype=int)

    for q in range(n):
        mlist = matches_per_query[q]
        k = min(topK, len(mlist))
        inl = np.array([float(mlist[i]["num_inliers"]) for i in range(k)], dtype=float)
        best_i = int(np.argmax(inl))
        reranked[q] = int(predictions_2d[q, best_i])
    return reranked


def recall_at_1_from_top1(top1_db_idx: np.ndarray, positives_per_query: list) -> float:
    n = len(top1_db_idx)
    hits = 0
    for q in range(n):
        pos = positives_per_query[q]
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().tolist()
        if int(top1_db_idx[q]) in set(pos):
            hits += 1
    return hits / n


# ----------------------------
# 3) Adaptive reranking
# ----------------------------

# if inliers < t then rerank
def adaptive_rerank_top1(predictions_2d: np.ndarray, reranked_top1: np.ndarray, inliers_top1: np.ndarray, threshold: float) -> np.ndarray:
    top1_retrieval = predictions_2d[:, 0].astype(int)
    use_rerank = inliers_top1 < threshold
    out = np.where(use_rerank, reranked_top1, top1_retrieval)
    return out.astype(int)


def sweep_thresholds(predictions_2d: np.ndarray, positives_per_query: list, reranked_top1: np.ndarray, inliers_top1: np.ndarray,thresholds: np.ndarray):
    r1_list = []
    frac_list = []

    # loops over many thresholds t
    for t in thresholds:
        top1_adapt = adaptive_rerank_top1(predictions_2d, reranked_top1, inliers_top1, t)
        r1 = recall_at_1_from_top1(top1_adapt, positives_per_query)
        frac = float(np.mean(inliers_top1 < t))
        r1_list.append(r1)
        frac_list.append(frac)

    return np.array(r1_list), np.array(frac_list)


# ----------------------------
# 4) Plots
# ----------------------------
def save_fig(plots_dir, name):
    if plots_dir is None:
        plt.show()
    else:
        os. makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, name), dpi=200, bbox_inches="tight")
        plt.close()

def plot_inliers_hist(plots_dir: str,inliers_top1: np.ndarray, correct: np.ndarray, title: str):
    correct_inl = inliers_top1[correct]
    wrong_inl = inliers_top1[~correct]

    plt.figure()
    plt.hist(correct_inl, bins=30, alpha=0.7, label=f"Correct @1 (n={len(correct_inl)})")
    plt.hist(wrong_inl, bins=30, alpha=0.7, label=f"Wrong @1 (n={len(wrong_inl)})")
    plt.xlabel("num_inliers (top-1 retrieved pair)")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    #plt.show()
    save_fig(plots_dir, f"{title}_hist.png")


def plot_sweep(plots_dir: str,
               thresholds: np.ndarray,
               r1: np.ndarray,
               frac_reranked: np.ndarray,
               title_prefix: str):
    plt.figure()
    plt.plot(thresholds, r1)
    plt.xlabel("inliers threshold t")
    plt.ylabel("Adaptive R@1")
    plt.title(f"{title_prefix} — R@1 vs threshold")
    #plt.show()
    save_fig(plots_dir,f"{title_prefix}_r1_vs_threshold.png")

    plt.figure()
    plt.plot(thresholds, 100 * frac_reranked)
    plt.xlabel("inliers threshold t")
    plt.ylabel("% queries reranked")
    plt.title(f"{title_prefix} — Rerank fraction vs threshold")
    #plt.show()
    save_fig(plots_dir,f"{title_prefix}_rerank_frac_vs_threshold.png")

def plot_time_tradeoff(plots_dir:str,
                       frac_reranked: np.ndarray,
                       t_rerank: float,
                       t_retrieval: float = 0.0,
                       title: str = "Time per query tradeoff"):
    T_always = t_retrieval + t_rerank
    T_adapt = t_retrieval + frac_reranked * t_rerank
    savings = 1.0 - (T_adapt / T_always)

    plt.figure()
    plt.plot(100 * frac_reranked, 100 * savings)
    plt.xlabel("% queries reranked")
    plt.ylabel("% time saved vs always-rerank")
    plt.title(title)
    #plt.show()
    
    save_fig(plots_dir,f"{title}_time_savings.png")


# ----------------------------
# 5) Main runner
# ----------------------------

def run_adaptive(z_path: str, matches_dir: str, dataset_name: str, t_rerank: float, t_retrieval: float,threshold_mode: str,num_thresholds: int, fixed_threshold: float, plots_dir: str):
    z = load_z_data(z_path)
    predictions = to_numpy_2d(z["predictions"])
    positives = z["positives_per_query"]

    num_q = predictions.shape[0]
    matches = load_matches_dir(matches_dir, expected_num_queries=num_q)

    if len(matches) != predictions.shape[0]:
        raise ValueError(
            f"Num match files ({len(matches)}) != num queries in predictions ({predictions.shape[0]}). "
            f"Ensure matches_dir has exactly one file per query in correct order."
        )

    correct = correct_at_1(predictions, positives)
    r1_retrieval = float(np.mean(correct))

    reranked_top1 = reranked_top1_from_inliers(predictions, matches)
    r1_reranked = recall_at_1_from_top1(reranked_top1, positives)

    inliers_top1 = get_inliers_top1(matches)

    if fixed_threshold is not None:
        top1_adapt = adaptive_rerank_top1(predictions, reranked_top1, inliers_top1, fixed_threshold)
        r1_adapt = recall_at_1_from_top1(top1_adapt, positives)
        frac = float(np.mean(inliers_top1 < fixed_threshold))

        print("\nFixed threshold evaluation:")
        print(f"  t = {fixed_threshold:.2f}")
        print(f"  Adaptive R@1 = {100*r1_adapt:.2f}")
        print(f"  % reranked   = {100*frac:.1f}")
        print(f"  Estimated time/query = {t_retrieval + frac*t_rerank:.3f}s (always-rerank: {t_retrieval + t_rerank:.3f}s)")
        return

    print(f"\n[{dataset_name}]")
    print(f"Retrieval-only R@1: {100*r1_retrieval:.2f}")
    print(f"Always-rerank  R@1: {100*r1_reranked:.2f}")
    print(f"Inliers(top1): min={inliers_top1.min():.1f}, median={np.median(inliers_top1):.1f}, max={inliers_top1.max():.1f}")

    plot_inliers_hist(plots_dir,inliers_top1, correct, title=f"{dataset_name}: inliers(top1) correct vs wrong")

    # threshold grid
    if threshold_mode == "quantiles":
        qs = np.linspace(0.0, 1.0, num_thresholds)
        thresholds = np.unique(np.quantile(inliers_top1, qs))
    elif threshold_mode == "range":
        lo = float(np.floor(inliers_top1.min()))
        hi = float(np.ceil(inliers_top1.max()))
        thresholds = np.linspace(lo, hi, num_thresholds)
    else:
        raise ValueError("threshold_mode must be one of: quantiles, range")

    r1_adapt, frac_reranked = sweep_thresholds(predictions, positives, reranked_top1, inliers_top1, thresholds)

    plot_sweep(plots_dir,thresholds, r1_adapt, frac_reranked, title_prefix=dataset_name)
    plot_time_tradeoff(plots_dir,frac_reranked, t_rerank=t_rerank, t_retrieval=t_retrieval, 
                       title=f"{dataset_name}: time savings (t_rerank={t_rerank}s)")

    best_idx = int(np.argmax(r1_adapt))
    best_t = float(thresholds[best_idx])
    best_r1 = float(r1_adapt[best_idx])
    best_frac = float(frac_reranked[best_idx])

    print("\nBest threshold (max R@1):")
    print(f"  t = {best_t:.2f}")
    print(f"  Adaptive R@1 = {100*best_r1:.2f}")
    print(f"  % reranked   = {100*best_frac:.1f}")
    print(f"  Estimated time/query = {t_retrieval + best_frac*t_rerank:.3f}s (always-rerank: {t_retrieval + t_rerank:.3f}s)")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Adaptive reranking using num_inliers on top-1.")
    p.add_argument("--z_path", type=str, required=True, help="Path to z_data.torch")
    p.add_argument("--matches_dir", type=str, required=True, help="Folder containing per-query match .torch files")
    p.add_argument("--dataset_name", type=str, default="Dataset")

    p.add_argument("--t_rerank", type=float, required=True, help="Reranking time per query for the matching model (seconds).")
    p.add_argument("--t_retrieval", type=float, default=0.0, help="Optional retrieval-only time per query (seconds).")

    p.add_argument("--threshold_mode", type=str, choices=["quantiles", "range"], default="quantiles",
                   help="How to create thresholds: quantiles or range.")
    p.add_argument("--num_thresholds", type=int, default=41, help="Number of thresholds to sweep (default 41).")
    p.add_argument("--fixed_threshold", type=float, default=None, help="If set, skip sweep and evaluate adaptive rerank using this threshold.")
    p.add_argument("--plots_dir", type=str, default=None, help="If set, save plots as PNGs into this folder.")
    
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    run_adaptive(
        z_path=args.z_path,
        matches_dir=args.matches_dir,
        dataset_name=args.dataset_name,
        t_rerank=args.t_rerank,
        t_retrieval=args.t_retrieval,
        threshold_mode=args.threshold_mode,
        num_thresholds=args.num_thresholds,
        fixed_threshold=args.fixed_threshold,
        plots_dir = args.plots_dir
    )
