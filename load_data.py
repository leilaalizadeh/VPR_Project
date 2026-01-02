import os
import re
import argparse
import torch
import numpy as np

import matplotlib.pyplot as plt


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

def to_numpy_2d(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    else:
        x = np.array(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
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

def recall_at_n(predictions_2d: np.ndarray, positives_per_query: list, N: int) -> float:
    """Recall@N: % queries having at least one positive in top-N predictions."""
    hits = 0
    Q = predictions_2d.shape[0]
    for q in range(Q):
        topN = predictions_2d[q, :N].astype(int).tolist()
        pos = positives_per_query[q]
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().tolist()
        pos = set(pos)
        if any(p in pos for p in topN):
            hits += 1
    return hits / Q

def reranked_preds_from_inliers(predictions_2d: np.ndarray, matches_per_query: list) -> np.ndarray:
    """
    Always-rerank list: reorder the retrieved top-K candidates by inliers descending.
    Returns QxK reordered predictions.
    """
    Q, K = predictions_2d.shape
    out = np.zeros_like(predictions_2d, dtype=int)

    for q in range(Q):
        mlist = matches_per_query[q]
        k_eff = min(K, len(mlist))
        inl = np.array([float(mlist[i]["num_inliers"]) for i in range(k_eff)], dtype=float)
        order = np.argsort(-inl)  # descending inliers
        out[q, :k_eff] = predictions_2d[q, order].astype(int)
        if k_eff < K:
            out[q, k_eff:] = predictions_2d[q, k_eff:].astype(int)

    return out