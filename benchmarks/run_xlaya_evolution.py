#!/usr/bin/env python3
"""
run_xlaya_evolution.py — Zero-Cheat Repo-Disjoint 3-Phase CUDA Sieve Evolution & Multi-Suite Verifier
Part of XLaya v2.3.0 (@iscvlab/xlaya, Apache-2.0).

Executes the 100% Zero-Leakage 4-Way Parity Evolution & Calibration Protocol:
  1. Loads repo-disjoint real `@base_commit` SWE issues (`SWE-bench_Lite dev` + disjoint `Loc-Bench_V1`)
     plus OOD negative counter-anchors (`When2Call train`), enforcing ZERO repository, commit, or
     instance_id overlap with `Kaggle 129` and `SWE-bench_Verified` via startup AssertionError.
  2. Extracts cross-regime `[M x 8]` AST/call-graph/Corpus-IDF features and smooth realized channel
     availability `M_avail` (`gamma_max = 1.45`), guaranteeing 4-way parity across CUDA evolution,
     Slot-1 Platt/conformal calibration, `laya_triage.py`, and Container A `sieve_oneshot.py`.
  3. Runs 5,000 generations x 256 genomes (1,280,000 evaluations) across:
       - Phase 1: Pure E4B-Sniper (`K=3`, `<= 800` tokens)
       - Phase 2: Conditional Residual 31B-Deep (`3` anchors + `5` residual halo slots, universal floors)
       - Phase 3: Teacher-Distilled E4B-Sniper (`K=3`, `<= 800` tokens)
  4. Calibrates L2-regularized Platt `(T, b)` (`T >= 0.08`) strictly on true Slot-1 gold-hit labels
     `y_i = 1{gold_count > 0 and file(slot_1) in gold_files}`, fits per-genome `conformal_q_hat`,
     calibrates the 2-Stage `When2Call` `tool_gate` strictly on `When2Call train`, verifies `--profile nano`
     subprocess RSS/latency + Suite 1/3 non-regression, and exports `policies/sieve_genomes.json` (< 15 KB).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
DATA_DIR = BENCH_DIR / "data"
POLICIES_DIR = REPO_ROOT / "policies"
SKILL_RES_DIR = REPO_ROOT / "skills" / "xlaya" / "resources"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import laya_triage as lt  # noqa: E402

# Hard-locked blind evaluation repositories (100% forbidden during evolution & calibration)
FORBIDDEN_BLIND_EVAL_REPOS = frozenset(
    {
        # Official Kaggle Gemma 4 Developer Agent 129-task repositories
        "encode/httpx",
        "fastapi/fastapi",
        "psf/requests",
        "textualize/rich",
        # Official SWE-bench_Verified 500-task repositories
        "astropy/astropy",
        "django/django",
        "matplotlib/matplotlib",
        "mwaskom/seaborn",
        "pallets/flask",
        "pydata/xarray",
        "pylint-dev/pylint",
        "pytest-dev/pytest",
        "scikit-learn/scikit-learn",
        "sphinx-doc/sphinx",
        "sympy/sympy",
    }
)


def _is_test_relpath(rel_path: str) -> bool:
    norm = "/" + rel_path.replace("\\", "/").lower().lstrip("/")
    fname = norm.rsplit("/", 1)[-1]
    return (
        fname.startswith("test_")
        or fname.endswith("_test.py")
        or any(mk in norm for mk in lt._TEST_PATH_MARKERS)
    )


def compute_ece_10bin(probs: Sequence[float], labels: Sequence[float]) -> float:
    if not probs:
        return 0.0
    p_arr = np.asarray(probs, dtype=np.float64)
    y_arr = np.asarray(labels, dtype=np.float64)
    ece = 0.0
    for b_idx in range(10):
        lo = b_idx / 10.0
        hi = (b_idx + 1) / 10.0
        mask = (p_arr >= lo) & (p_arr < hi if b_idx < 9 else p_arr <= hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(p_arr[mask])) - float(np.mean(y_arr[mask])))
    return round(float(ece), 4)


def compute_brier(probs: Sequence[float], labels: Sequence[float]) -> float:
    if not probs:
        return 0.0
    p_arr = np.asarray(probs, dtype=np.float64)
    y_arr = np.asarray(labels, dtype=np.float64)
    return round(float(np.mean((p_arr - y_arr) ** 2)), 4)


def verify_manifest_zero_leakage(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Assert 0% repo, commit, or task overlap with Kaggle 129 and SWE-bench_Verified."""
    tasks = manifest.get("tasks") or []
    if not tasks:
        raise AssertionError("Disjoint manifest contains 0 tasks.")

    train_repos: set[str] = set()
    val_repos: set[str] = set()
    seen_ids: set[str] = set()

    for t in tasks:
        tid = str(t.get("id", ""))
        repo = str(t.get("repo", ""))
        split = str(t.get("split", ""))
        if not tid or not repo or split not in ("train", "val"):
            raise AssertionError(f"Malformed manifest task record: {t}")
        if tid in seen_ids:
            raise AssertionError(f"Duplicate task id in manifest: {tid}")
        seen_ids.add(tid)
        if repo in FORBIDDEN_BLIND_EVAL_REPOS or repo.lower() in {r.lower() for r in FORBIDDEN_BLIND_EVAL_REPOS}:
            raise AssertionError(f"DATA LEAKAGE DETECTED: Forbidden blind-eval repo {repo!r} in manifest!")
        if split == "train":
            train_repos.add(repo)
        else:
            val_repos.add(repo)

    overlap_tv = train_repos & val_repos
    if overlap_tv:
        raise AssertionError(f"Train/Val repository leakage detected: {sorted(overlap_tv)}")

    pos_cnt = sum(1 for t in tasks if t.get("gold_files"))
    neg_cnt = sum(1 for t in tasks if not t.get("gold_files"))
    return {
        "zero_leakage_verified": True,
        "total_tasks": len(tasks),
        "positive_tasks": pos_cnt,
        "negative_ood_tasks": neg_cnt,
        "train_repos_count": len(train_repos),
        "val_repos_count": len(val_repos),
        "forbidden_repos_blocked": len(FORBIDDEN_BLIND_EVAL_REPOS),
    }


def build_disjoint_swe_feature_dataset(
    manifest_path: Path,
    *,
    max_chunks: int = 4096,
) -> Dict[str, Any]:
    """Build frozen cross-regime `[M x 8]` + `M_avail` dataset from verified disjoint `@base_commit` snapshots."""
    t0 = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leak_audit = verify_manifest_zero_leakage(manifest)

    # Cache file lists and per-file AST chunks per snapshot_slug so shared commits parse AST once
    snap_files_cache: Dict[str, Tuple[List[Path], Path]] = {}
    file_chunks_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    records: List[Dict[str, Any]] = []
    snap_py_counts: Dict[str, int] = {}
    task_chunk_counts: List[int] = []

    for spec in manifest["tasks"]:
        slug = str(spec["snapshot_slug"])
        snap_dir = Path(str(spec["snapshot_dir"]))
        if not snap_dir.is_dir():
            raise RuntimeError(f"Snapshot directory missing for {spec['id']}: {snap_dir}")

        if slug not in snap_files_cache:
            files_raw, s_root = lt.collect_target_files([str(snap_dir)], pattern="*.py,*.pyi")
            snap_files_cache[slug] = (files_raw, s_root or snap_dir)
            snap_py_counts[slug] = len(files_raw)

        base_files, s_root = snap_files_cache[slug]
        ranked_files = lt._prerank_repo_files(base_files, s_root, spec["query"])

        chunks: List[Dict[str, Any]] = []
        for fp in ranked_files:
            rel_pfx = fp.relative_to(s_root).as_posix()
            ck_key = (slug, rel_pfx)
            if ck_key not in file_chunks_cache:
                extracted = lt.chunk_file(fp, max_chars=1600, rel_prefix=rel_pfx)
                for ch in extracted:
                    ch["file"] = f"{spec['repo']}/{rel_pfx}"
                    ch["repo_rel"] = rel_pfx
                file_chunks_cache[ck_key] = extracted
            rem = max_chunks - len(chunks)
            if rem <= 0:
                break
            chunks.extend(file_chunks_cache[ck_key][:rem])

        feats_list = lt.extract_code_graph_features(chunks, spec["query"])
        m_avail_list = lt.compute_channel_availability(chunks, feats_list)
        feats = np.asarray(feats_list, dtype=np.float32)
        m_avail = np.asarray(m_avail_list, dtype=np.float32)

        file_names = [c["repo_rel"] for c in chunks]
        unique_files = list(dict.fromkeys(file_names))
        gold_files = [str(gf).replace("\\", "/") for gf in (spec.get("gold_files") or [])]
        # Ensure all gold files have a valid file ID even if outside max_chunks so recall denominator is exact
        for gf in gold_files:
            if gf not in unique_files:
                unique_files.append(gf)

        f2id = {fn: i for i, fn in enumerate(unique_files)}
        chunk_file_ids = np.asarray([f2id[fn] for fn in file_names], dtype=np.int32)
        is_test_chunk = np.asarray([1.0 if _is_test_relpath(fn) else 0.0 for fn in file_names], dtype=np.float32)
        est_tokens = np.asarray([lt._chunk_total_est_tokens(c) for c in chunks], dtype=np.int32)
        gold_set = set(gold_files)
        is_gold_chunk = np.asarray([1.0 if fn in gold_set else 0.0 for fn in file_names], dtype=np.float32)
        task_chunk_counts.append(len(chunks))

        # Keep lightweight chunk descriptors for exact `_select_sieve_slots` parity
        chunks_slim = [
            {
                "file": c["file"],
                "repo_rel": c["repo_rel"],
                "name": c["name"],
                "ref": c["ref"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "text": c["text"],
                "_ast_meta": c.get("_ast_meta"),
            }
            for c in chunks
        ]

        records.append(
            {
                "id": str(spec["id"]),
                "repo": str(spec["repo"]),
                "split": str(spec["split"]),
                "multi_hop": bool(spec.get("multi_hop", len(gold_files) > 1)),
                "query": str(spec["query"]),
                "gold_files": gold_files,
                "gold_file_ids": [f2id[gf] for gf in gold_files],
                "gold_count": len(gold_files),
                "features": feats,
                "features_list": feats_list,
                "m_avail": m_avail,
                "m_avail_list": m_avail_list,
                "chunks_slim": chunks_slim,
                "chunk_file_ids": chunk_file_ids,
                "is_test_chunk": is_test_chunk,
                "is_gold_chunk": is_gold_chunk,
                "est_tokens": est_tokens,
                "unique_files": unique_files,
            }
        )

    elapsed_s = round(time.perf_counter() - t0, 3)
    return {
        "leak_audit": leak_audit,
        "records": records,
        "snapshot_count": len(snap_py_counts),
        "total_py_files": sum(snap_py_counts.values()),
        "total_chunks_evaluated": sum(task_chunk_counts),
        "mean_chunks_per_task": round(sum(task_chunk_counts) / max(1, len(task_chunk_counts)), 1),
        "feature_extract_sec": elapsed_s,
    }


def evaluate_genome_on_records(
    records: Sequence[Mapping[str, Any]],
    w_core: Sequence[float],
    *,
    top_k: int,
    max_tokens: int,
    w_residual: Optional[Sequence[float]] = None,
    anchor_slots: int = 3,
    diversity_lambda: float = 0.25,
    platt_T: float = 0.14,
    platt_b: float = -0.28,
    conformal_q_hat: float = 0.62,
    gamma_max: float = 1.45,
) -> Dict[str, Any]:
    """
    Evaluate an 8-weight genome using `lt._select_sieve_slots` directly so evolution/calibration
    is 100% bit-exact with `laya_triage.sieve_code_graph` (`M_avail`, `_span_iou <= 0.30`, budget wall).
    """
    agent_tier = "31b-deep" if w_residual is not None else "e4b-sniper"
    g_cfg: Dict[str, Any] = {
        "w_core": list(w_core),
        "top_k": top_k,
        "token_budget": max_tokens,
        "lambda_file_div": diversity_lambda,
        "platt_T": platt_T,
        "platt_b": platt_b,
        "conformal_q_hat": conformal_q_hat,
        "gamma_max": gamma_max,
        "anchor_slots": anchor_slots,
    }
    if w_residual is not None:
        g_cfg["w_halo"] = list(w_residual)

    pos_recalls: List[float] = []
    pos_hit_any: List[float] = []
    multihop_recalls: List[float] = []
    pos_precisions: List[float] = []
    pos_mrrs: List[float] = []
    test_echo_rates: List[float] = []
    tokens_used_list: List[int] = []
    zero_safe_recalls_all: List[float] = []

    top1_probs_all: List[float] = []
    top1_labels_all: List[float] = []
    pos_top1_probs: List[float] = []
    neg_top1_probs: List[float] = []
    neg_wrong_confident_80 = 0
    neg_abstained = 0
    neg_total = 0

    per_task_selected_files: Dict[str, List[str]] = {}

    for rec in records:
        chunks = rec["chunks_slim"]
        sel_idxs, _core_sc, probs, used_tok = lt._select_sieve_slots(
            chunks,
            rec["features_list"],
            g_cfg,
            agent_tier=agent_tier,
            top_k=top_k,
            max_tokens=max_tokens,
            m_avail=rec["m_avail_list"],
        )
        tokens_used_list.append(used_tok)
        sel_files: List[str] = []
        for idx in sel_idxs:
            rf = chunks[idx]["repo_rel"]
            if rf not in sel_files:
                sel_files.append(rf)
        per_task_selected_files[rec["id"]] = sel_files

        top1_p = float(probs[sel_idxs[0]]) if sel_idxs else 0.0
        abstain = top1_p < (1.0 - conformal_q_hat)

        gold_set = set(rec["gold_files"])
        gold_count = int(rec["gold_count"])
        top1_file = chunks[sel_idxs[0]]["repo_rel"] if sel_idxs else ""
        y_top1 = 1.0 if (gold_count > 0 and top1_file in gold_set) else 0.0

        top1_probs_all.append(top1_p)
        top1_labels_all.append(y_top1)

        test_hits = sum(1 for ii in sel_idxs if float(rec["is_test_chunk"][ii]) > 0.5)
        test_echo_rates.append(test_hits / max(1, len(sel_idxs)))

        if gold_count == 0:
            neg_total += 1
            neg_top1_probs.append(top1_p)
            if abstain:
                neg_abstained += 1
            if top1_p >= 0.80 and not abstain:
                neg_wrong_confident_80 += 1
            zero_safe_recalls_all.append(1.0 if abstain else 0.0)
        else:
            pos_top1_probs.append(top1_p)
            hits = sum(1 for fn in sel_files if fn in gold_set)
            rec_k = hits / max(1, gold_count)
            prec_k = hits / max(1, len(sel_files))
            rr = 0.0
            for rank_idx, fn in enumerate(sel_files, start=1):
                if fn in gold_set:
                    rr = 1.0 / rank_idx
                    break
            pos_recalls.append(rec_k)
            pos_hit_any.append(1.0 if hits > 0 else 0.0)
            if rec["multi_hop"]:
                multihop_recalls.append(rec_k)
            pos_precisions.append(prec_k)
            pos_mrrs.append(rr)
            zero_safe_recalls_all.append(rec_k)

    pos_idx_mask = [i for i, r in enumerate(records) if int(r["gold_count"]) > 0]
    pos_probs_only = [top1_probs_all[i] for i in pos_idx_mask]
    pos_labels_only = [top1_labels_all[i] for i in pos_idx_mask]

    return {
        "pos_file_recall_at_k": round(float(np.mean(pos_recalls)) if pos_recalls else 0.0, 4),
        "pos_task_hit_rate_at_k": round(float(np.mean(pos_hit_any)) if pos_hit_any else 0.0, 4),
        "multihop_recall_at_k": round(float(np.mean(multihop_recalls)) if multihop_recalls else 0.0, 4),
        "pos_file_precision_at_k": round(float(np.mean(pos_precisions)) if pos_precisions else 0.0, 4),
        "pos_mrr": round(float(np.mean(pos_mrrs)) if pos_mrrs else 0.0, 4),
        "zero_safe_overall_score": round(float(np.mean(zero_safe_recalls_all)) if zero_safe_recalls_all else 0.0, 4),
        "test_echo_rate": round(float(np.mean(test_echo_rates)) if test_echo_rates else 0.0, 4),
        "mean_selected_tokens": round(float(np.mean(tokens_used_list)) if tokens_used_list else 0.0, 1),
        "top1_ece_all": compute_ece_10bin(top1_probs_all, top1_labels_all),
        "top1_ece_pos": compute_ece_10bin(pos_probs_only, pos_labels_only),
        "top1_brier_all": compute_brier(top1_probs_all, top1_labels_all),
        "neg_abstain_rate": round(neg_abstained / max(1, neg_total), 4),
        "neg_wrong_confident_80_rate": round(neg_wrong_confident_80 / max(1, neg_total), 4),
        "prob_min": round(min(top1_probs_all) if top1_probs_all else 0.0, 4),
        "prob_max": round(max(top1_probs_all) if top1_probs_all else 0.0, 4),
        "pos_prob_max": round(max(pos_top1_probs) if pos_top1_probs else 0.0, 4),
        "neg_prob_max": round(max(neg_top1_probs) if neg_top1_probs else 0.0, 4),
        "per_task_selected_files": per_task_selected_files,
    }


def _pack_split_tensors(records: Sequence[Mapping[str, Any]], device_str: str) -> Dict[str, Any]:
    import torch

    dev = torch.device(device_str)
    n_tasks = len(records)
    m_max = max(r["features"].shape[0] for r in records)
    f_max = max(len(r["unique_files"]) for r in records)

    feat_np = np.zeros((n_tasks, m_max, 8), dtype=np.float32)
    m_avail_np = np.ones((n_tasks, 8), dtype=np.float32)
    valid_np = np.zeros((n_tasks, m_max), dtype=np.bool_)
    chunk_fid_np = np.zeros((n_tasks, m_max), dtype=np.int64)
    file_gold_np = np.zeros((n_tasks, f_max), dtype=np.float32)
    file_test_np = np.zeros((n_tasks, f_max), dtype=np.float32)
    file_valid_np = np.zeros((n_tasks, f_max), dtype=np.bool_)
    is_pos_np = np.zeros((n_tasks,), dtype=np.bool_)
    is_mh_np = np.zeros((n_tasks,), dtype=np.bool_)
    gold_cnt_np = np.ones((n_tasks,), dtype=np.float32)

    for i, r in enumerate(records):
        m = r["features"].shape[0]
        nf = len(r["unique_files"])
        feat_np[i, :m, :] = r["features"]
        m_avail_np[i, :] = r["m_avail"]
        valid_np[i, :m] = True
        chunk_fid_np[i, :m] = r["chunk_file_ids"]
        is_pos_np[i] = int(r["gold_count"]) > 0
        is_mh_np[i] = bool(r["multi_hop"]) and int(r["gold_count"]) > 0
        gold_cnt_np[i] = float(max(1, int(r["gold_count"])))
        file_valid_np[i, :nf] = True
        for gf_id in r["gold_file_ids"]:
            file_gold_np[i, gf_id] = 1.0
        for fid_idx, fname in enumerate(r["unique_files"]):
            if _is_test_relpath(fname):
                file_test_np[i, fid_idx] = 1.0

    return {
        "F": torch.from_numpy(feat_np).to(dev),
        "m_avail": torch.from_numpy(m_avail_np).to(dev),
        "valid": torch.from_numpy(valid_np).to(dev),
        "chunk_fid": torch.from_numpy(chunk_fid_np).to(dev),
        "file_gold": torch.from_numpy(file_gold_np).to(dev),
        "file_test": torch.from_numpy(file_test_np).to(dev),
        "file_valid": torch.from_numpy(file_valid_np).to(dev),
        "is_pos": torch.from_numpy(is_pos_np).to(dev),
        "is_mh": torch.from_numpy(is_mh_np).to(dev),
        "gold_cnt": torch.from_numpy(gold_cnt_np).to(dev),
        "f_max": f_max,
    }


def _project_pop_simplex_torch(W_raw: Any, *, is_halo: bool = False) -> Any:
    """
    Sign-constrained L1 simplex projection with universal channel floors (`G-01`, `G-03`, `G-08`):
      - `w_0 >= 0.05` (lexical safety floor)
      - `w_5 >= 0.06` (halo) / `0.045` (core) (`f_5_tfidf` / `.npz` seed-propagation floor)
      - `w_7 >= 0.04` (stack-trace / exception line boost floor)
      - `w_6 in [-0.40, -0.06]` (test-echo penalty)
      - `||W||_1 == 1.0`
    """
    import torch

    pos_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1], dtype=W_raw.dtype, device=W_raw.device).unsqueeze(1)
    neg_mask = 1.0 - pos_mask
    min_pos = torch.tensor(
        [0.055, 0.02, 0.02, 0.03, 0.015, 0.068 if is_halo else 0.050, 0.0, 0.045],
        dtype=W_raw.dtype,
        device=W_raw.device,
    ).unsqueeze(1)

    W = W_raw
    for _ in range(2):
        w_pos = torch.maximum(torch.clamp(W, min=0.0), min_pos) * pos_mask
        w_neg = torch.clamp(W, min=-0.40, max=-0.065) * neg_mask
        W = w_pos + w_neg
        l1 = torch.sum(torch.abs(W), dim=0, keepdim=True).clamp(min=1e-8)
        W = W / l1
    return W


def _apply_active_simplex_torch(W: Any, m_avail_mat: Any, gamma_max: float = 1.45) -> Any:
    """Exact tensor counterpart to `lt.apply_active_simplex_weights`: returns `W_eff` of shape `[N, 8, P]`."""
    import torch

    pos_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 1], dtype=W.dtype, device=W.device).view(1, 8, 1)
    w_pos = torch.clamp(W.unsqueeze(0), min=0.0) * pos_mask  # [1, 8, P]
    w_pos_total = w_pos.sum(dim=1, keepdim=True)  # [1, 1, P]
    m_exp = m_avail_mat.unsqueeze(-1)  # [N, 8, 1]
    active_mass = (w_pos * m_exp).sum(dim=1, keepdim=True)  # [N, 1, P]
    scale = torch.clamp(w_pos_total / active_mass.clamp(min=1e-8), max=float(gamma_max))  # [N, 1, P]
    W_eff = w_pos * m_exp * scale  # [N, 8, P]
    W_eff[:, 6, :] = W[6, :].unsqueeze(0)
    return W_eff


def _eval_pop_file_objective_torch(
    pack: Dict[str, Any],
    W: Any,
    *,
    top_k_files: int,
    anchor_file_mask: Optional[Any] = None,
    teacher_soft_targets: Optional[Any] = None,
    gamma_max: float = 1.45,
) -> Any:
    import torch

    F_mat = pack["F"]  # [N, M, 8]
    valid = pack["valid"].unsqueeze(-1)  # [N, M, 1]
    n_tasks, _, _ = F_mat.shape
    pop = W.shape[1]
    f_max = pack["f_max"]

    W_eff = _apply_active_simplex_torch(W, pack["m_avail"], gamma_max=gamma_max)  # [N, 8, P]
    chunk_scores = torch.einsum("nmd,ndp->nmp", F_mat, W_eff)  # [N, M, P]
    chunk_scores = chunk_scores.masked_fill(~valid, -1e4)

    fid_idx = pack["chunk_fid"].unsqueeze(-1).expand(n_tasks, chunk_scores.shape[1], pop)
    file_scores = torch.full((n_tasks, f_max, pop), -1e4, dtype=chunk_scores.dtype, device=W.device)
    file_scores.scatter_reduce_(1, fid_idx, chunk_scores, reduce="amax", include_self=True)

    f_valid = pack["file_valid"].unsqueeze(-1)
    file_scores = file_scores.masked_fill(~f_valid, -1e4)

    masked_scores = file_scores.masked_fill(anchor_file_mask, -1e4) if anchor_file_mask is not None else file_scores
    k_eff = min(top_k_files, f_max)
    _, topk_fids = torch.topk(masked_scores, k=k_eff, dim=1)

    gold_exp = pack["file_gold"].unsqueeze(-1).expand(n_tasks, f_max, pop)
    test_exp = pack["file_test"].unsqueeze(-1).expand(n_tasks, f_max, pop)
    hits_at_k = torch.gather(gold_exp, 1, topk_fids)
    tests_at_k = torch.gather(test_exp, 1, topk_fids)

    if anchor_file_mask is not None:
        anc_hits = (pack["file_gold"].unsqueeze(-1) * anchor_file_mask.float()).sum(dim=1)
        total_hits = anc_hits + hits_at_k.sum(dim=1)
        denom_k = float(k_eff + 3)
    else:
        total_hits = hits_at_k.sum(dim=1)
        denom_k = float(k_eff)

    pos_mask = pack["is_pos"]
    mh_mask = pack["is_mh"]
    neg_mask = ~pos_mask

    rec_per_task = total_hits / pack["gold_cnt"].unsqueeze(-1)
    prec_per_task = total_hits / max(1.0, denom_k)
    ranks = torch.arange(1, k_eff + 1, dtype=chunk_scores.dtype, device=W.device).view(1, k_eff, 1)
    weighted_rr = (hits_at_k / ranks).max(dim=1).values

    pos_rec = rec_per_task[pos_mask].mean(dim=0)
    mh_rec = rec_per_task[mh_mask].mean(dim=0) if torch.any(mh_mask) else pos_rec
    pos_prec = prec_per_task[pos_mask].mean(dim=0)
    pos_rr = weighted_rr[pos_mask].mean(dim=0)
    test_echo = tests_at_k.mean(dim=1).mean(dim=0)

    top1_raw = file_scores.max(dim=1).values
    pos_top1 = top1_raw[pos_mask].mean(dim=0)
    neg_top1 = top1_raw[neg_mask].max(dim=0).values if torch.any(neg_mask) else (pos_top1 * 0.25)
    margin_sep = pos_top1 - neg_top1

    if anchor_file_mask is not None:
        return 0.46 * mh_rec + 0.34 * pos_rec + 0.14 * pos_rr + 0.12 * margin_sep - 0.45 * test_echo
    if teacher_soft_targets is not None:
        student_prob = torch.softmax(file_scores / 0.12, dim=1)
        kl_align = (teacher_soft_targets * torch.log(student_prob.clamp(min=1e-7))).sum(dim=1)[pos_mask].mean(dim=0)
        return (
            0.34 * pos_prec
            + 0.36 * pos_rec
            + 0.20 * pos_rr
            + 0.14 * margin_sep
            + 0.08 * kl_align
            - 0.50 * test_echo
        )
    return 0.36 * pos_prec + 0.34 * pos_rec + 0.22 * pos_rr + 0.14 * margin_sep - 0.50 * test_echo


def calibrate_platt_and_conformal(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    w_core: Sequence[float],
    *,
    top_k: int = 3,
    max_tokens: int = 800,
    w_residual: Optional[Sequence[float]] = None,
    diversity_lambda: float = 0.25,
    gamma_max: float = 1.45,
) -> Tuple[float, float, float]:
    """
    Fit L2-regularized Platt `(T, b)` (`T >= 0.08`) strictly on **Selected Slot-1 True-Hit Labels** (`F-06`):
      `y_i = 1{gold_count > 0 and file(slot_1) in gold_files}`
    and fit per-genome `conformal_q_hat` on `val_records` preserving 0% OOD false-confident fires.
    """
    agent_tier = "31b-deep" if w_residual is not None else "e4b-sniper"
    g_cfg: Dict[str, Any] = {
        "w_core": list(w_core),
        "top_k": top_k,
        "token_budget": max_tokens,
        "lambda_file_div": diversity_lambda,
        "platt_T": 0.15,
        "platt_b": -0.25,
        "gamma_max": gamma_max,
    }
    if w_residual is not None:
        g_cfg["w_halo"] = list(w_residual)

    w_static = lt.project_sieve_weights_simplex(w_core)

    def _collect_slot1(recs: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_list: List[float] = []
        y_list: List[float] = []
        is_neg_list: List[bool] = []
        for r in recs:
            chunks = r["chunks_slim"]
            feats_mat = r["features_list"]
            sel_idxs, core_sc, _, _ = lt._select_sieve_slots(
                chunks,
                feats_mat,
                g_cfg,
                agent_tier=agent_tier,
                top_k=top_k,
                max_tokens=max_tokens,
                m_avail=r["m_avail_list"],
            )
            if sel_idxs:
                idx0 = sel_idxs[0]
                st0 = sum(float(feats_mat[idx0][d]) * w_static[d] for d in range(8))
                s1 = 0.35 * float(core_sc[idx0]) + 0.65 * st0
            else:
                s1 = 0.0
            top1_file = chunks[sel_idxs[0]]["repo_rel"] if sel_idxs else ""
            gold_cnt = int(r["gold_count"])
            y1 = 1.0 if (gold_cnt > 0 and top1_file in set(r["gold_files"])) else 0.0
            s_list.append(s1)
            y_list.append(y1)
            is_neg_list.append(gold_cnt == 0)
        return (
            np.asarray(s_list, dtype=np.float64),
            np.asarray(y_list, dtype=np.float64),
            np.asarray(is_neg_list, dtype=np.bool_),
        )

    tr_s, tr_y, tr_neg = _collect_slot1(train_records)
    va_s, va_y, va_neg = _collect_slot1(val_records)
    va_pos = ~va_neg

    best_obj = 1e9
    best_T, best_offset = 1.636, 0.8016

    for offset_cand in np.linspace(0.795, 0.808, 15):
        for t_cand in np.linspace(1.60, 1.67, 15):
            logits_tr = np.clip((tr_s - offset_cand) / t_cand, -30.0, 30.0)
            p_tr = 1.0 / (1.0 + np.exp(-logits_tr))
            if np.any(tr_neg) and float(np.max(p_tr[tr_neg])) >= 0.72:
                continue
            eps = 1e-7
            nll_tr = -float(np.mean(tr_y * np.log(p_tr + eps) + (1.0 - tr_y) * np.log(1.0 - p_tr + eps)))
            brier_tr = float(np.mean((p_tr - tr_y) ** 2))
            l2_reg = 0.015 * ((1.0 / t_cand - 0.61) ** 2 / 25.0)

            logits_va = np.clip((va_s - offset_cand) / t_cand, -30.0, 30.0)
            p_va = 1.0 / (1.0 + np.exp(-logits_va))
            if np.any(va_neg) and float(np.max(p_va[va_neg])) >= 0.72:
                continue
            p_va_pos = p_va[va_pos] if np.any(va_pos) else p_va
            y_va_pos = va_y[va_pos] if np.any(va_pos) else va_y
            ece_va_pos = compute_ece_10bin(p_va_pos.tolist(), y_va_pos.tolist())
            ece_va_all = compute_ece_10bin(p_va.tolist(), va_y.tolist())
            mean_gap = abs(float(np.mean(p_va)) - float(np.mean(va_y)))
            obj = 0.15 * nll_tr + 0.30 * ece_va_pos + 0.35 * ece_va_all + 0.13 * mean_gap + 0.07 * brier_tr + l2_reg
            if obj < best_obj:
                best_obj = obj
                best_T, best_offset = float(t_cand), float(offset_cand)

    va_logits = np.clip((va_s - best_offset) / best_T, -30.0, 30.0)
    va_probs = 1.0 / (1.0 + np.exp(-va_logits))
    tr_logits = np.clip((tr_s - best_offset) / best_T, -30.0, 30.0)
    tr_probs = 1.0 / (1.0 + np.exp(-tr_logits))

    # Conformal abstain threshold `1 - q_hat`: bound strictly above OOD negative quantiles while
    # avoiding over-abstention on genuine symptom-only SWE bug reports (`F-06`).
    all_neg_probs = (
        np.concatenate([tr_probs[tr_neg], va_probs[va_neg]])
        if (np.any(tr_neg) or np.any(va_neg))
        else np.array([0.18])
    )
    neg_q90 = float(np.percentile(all_neg_probs, 90))
    pos_hit_probs = np.concatenate([tr_probs[tr_y > 0.5], va_probs[va_y > 0.5]])
    if pos_hit_probs.size > 0:
        nonconf = np.sort(1.0 - pos_hit_probs)
        idx_q = min(nonconf.size - 1, int(math.ceil(0.90 * (nonconf.size + 1))) - 1)
        cand_thresh = 1.0 - float(nonconf[max(0, idx_q)])
    else:
        cand_thresh = 0.34

    abstain_thresh = max(0.24, min(0.4115, max(neg_q90 + 0.002, cand_thresh)))
    q_hat = 1.0 - abstain_thresh
    return round(best_T, 4), round(-best_offset, 4), round(q_hat, 4)


def run_three_phase_cuda_evolution(
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    *,
    total_generations: int = 5000,
    pop_size: int = 256,
    seed: int = 42,
    gamma_max: float = 1.45,
) -> Dict[str, Any]:
    import torch

    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if dev_str == "cuda" else "CPU"
    torch.manual_seed(seed)
    if dev_str == "cuda":
        torch.cuda.manual_seed_all(seed)

    tr_pack = _pack_split_tensors(train_records, dev_str)
    va_pack = _pack_split_tensors(val_records, dev_str)

    g_p1 = int(round(total_generations * 0.36))
    g_p2 = int(round(total_generations * 0.36))
    g_p3 = max(1, total_generations - g_p1 - g_p2)

    prior_sniper = torch.tensor(
        [0.24, 0.16, 0.16, 0.11, 0.05, 0.10, -0.10, 0.08],
        dtype=torch.float32,
        device=dev_str,
    ).unsqueeze(1)

    def _evolve_phase(
        init_center: Any,
        n_gens: int,
        top_k_files: int,
        *,
        is_halo: bool = False,
        anchor_mask_tr: Optional[Any] = None,
        anchor_mask_va: Optional[Any] = None,
        teacher_tr: Optional[Any] = None,
        teacher_va: Optional[Any] = None,
    ) -> Tuple[List[float], float, float]:
        elite_k = max(8, pop_size // 8)
        pop = init_center.repeat(1, pop_size)
        noise0 = torch.randn_like(pop) * 0.075
        noise0[:, 0] = 0.0
        pop = _project_pop_simplex_torch(pop + noise0, is_halo=is_halo)

        best_val_fit = -1e9
        best_tr_fit = -1e9
        best_w_vec: Optional[Any] = None

        for gen in range(n_gens):
            sigma = 0.060 * (1.0 - 0.75 * (gen / max(1, n_gens - 1)))
            fit_tr = _eval_pop_file_objective_torch(
                tr_pack,
                pop,
                top_k_files=top_k_files,
                anchor_file_mask=anchor_mask_tr,
                teacher_soft_targets=teacher_tr,
                gamma_max=gamma_max,
            )
            elite_vals, elite_idxs = torch.topk(fit_tr, k=elite_k)
            elites = pop[:, elite_idxs]

            if (gen % 25 == 0) or (gen == n_gens - 1):
                fit_va = _eval_pop_file_objective_torch(
                    va_pack,
                    elites,
                    top_k_files=top_k_files,
                    anchor_file_mask=anchor_mask_va,
                    teacher_soft_targets=teacher_va,
                    gamma_max=gamma_max,
                )
                va_max, va_arg = torch.max(fit_va, dim=0)
                cand_score = float(va_max.item()) + 0.05 * float(elite_vals[va_arg].item())
                if cand_score > best_val_fit:
                    best_val_fit = cand_score
                    best_tr_fit = float(elite_vals[va_arg].item())
                    best_w_vec = elites[:, int(va_arg.item())].detach().clone()

            n_children = pop_size - elite_k
            p_a = elites[:, torch.randint(0, elite_k, (n_children,), device=dev_str)]
            p_b = elites[:, torch.randint(0, elite_k, (n_children,), device=dev_str)]
            alpha = torch.rand((1, n_children), device=dev_str)
            children = alpha * p_a + (1.0 - alpha) * p_b
            mut = torch.randn((8, n_children), device=dev_str) * sigma
            children = _project_pop_simplex_torch(children + mut, is_halo=is_halo)
            pop = torch.cat([elites, children], dim=1)

        assert best_w_vec is not None
        w_list = lt.project_sieve_weights_simplex(best_w_vec.cpu().tolist())
        return w_list, round(best_tr_fit, 4), round(best_val_fit, 4)

    if dev_str == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    w_p1, tr_p1, va_p1 = _evolve_phase(prior_sniper, g_p1, top_k_files=3, is_halo=False)
    w_p1_t = torch.tensor(w_p1, dtype=torch.float32, device=dev_str).unsqueeze(1)

    def _compute_anchor_mask_and_teacher(pack: Dict[str, Any], w_col: Any, k_files: int) -> Tuple[Any, Any]:
        w_eff = _apply_active_simplex_torch(w_col, pack["m_avail"], gamma_max=gamma_max)
        cs = torch.einsum("nmd,ndp->nmp", pack["F"], w_eff).masked_fill(~pack["valid"].unsqueeze(-1), -1e4)
        fs = torch.full((pack["F"].shape[0], pack["f_max"], 1), -1e4, dtype=cs.dtype, device=dev_str)
        fs.scatter_reduce_(1, pack["chunk_fid"].unsqueeze(-1), cs, reduce="amax", include_self=True)
        fs = fs.masked_fill(~pack["file_valid"].unsqueeze(-1), -1e4)
        _, tk_fids = torch.topk(fs, k=min(k_files, pack["f_max"]), dim=1)
        mask = torch.zeros((pack["F"].shape[0], pack["f_max"], 1), dtype=torch.bool, device=dev_str)
        mask.scatter_(1, tk_fids, True)
        return mask, fs

    anc_mask_tr, fs_anc_tr = _compute_anchor_mask_and_teacher(tr_pack, w_p1_t, 3)
    anc_mask_va, fs_anc_va = _compute_anchor_mask_and_teacher(va_pack, w_p1_t, 3)

    prior_deep_res = torch.tensor(
        [0.14, 0.14, 0.13, 0.19, 0.10, 0.14, -0.10, 0.06],
        dtype=torch.float32,
        device=dev_str,
    ).unsqueeze(1)
    warm_p2 = _project_pop_simplex_torch(0.40 * w_p1_t + 0.60 * prior_deep_res, is_halo=True)
    w_p2_res, tr_p2, va_p2 = _evolve_phase(
        warm_p2,
        g_p2,
        top_k_files=5,
        is_halo=True,
        anchor_mask_tr=anc_mask_tr,
        anchor_mask_va=anc_mask_va,
    )

    w_p2_t = torch.tensor(w_p2_res, dtype=torch.float32, device=dev_str).unsqueeze(1)
    _, fs_res_tr = _compute_anchor_mask_and_teacher(tr_pack, w_p2_t, 8)
    _, fs_res_va = _compute_anchor_mask_and_teacher(va_pack, w_p2_t, 8)
    teacher_tr = torch.softmax((0.5 * fs_anc_tr + 0.5 * fs_res_tr) / 0.12, dim=1)
    teacher_va = torch.softmax((0.5 * fs_anc_va + 0.5 * fs_res_va) / 0.12, dim=1)

    warm_p3 = _project_pop_simplex_torch(0.70 * w_p1_t + 0.30 * w_p2_t, is_halo=False)
    w_p3_distilled, tr_p3, va_p3 = _evolve_phase(
        warm_p3,
        g_p3,
        top_k_files=3,
        is_halo=False,
        teacher_tr=teacher_tr,
        teacher_va=teacher_va,
    )

    if dev_str == "cuda":
        torch.cuda.synchronize()
    evolve_elapsed_s = round(time.perf_counter() - t_start, 4)

    best_div_lam = 0.24
    best_div_score = -1.0
    for cand_lam in (0.16, 0.22, 0.28, 0.34):
        va_m = evaluate_genome_on_records(
            val_records,
            w_p3_distilled,
            top_k=8,
            max_tokens=4096,
            w_residual=w_p2_res,
            anchor_slots=3,
            diversity_lambda=cand_lam,
            gamma_max=gamma_max,
        )
        sc = 0.55 * va_m["multihop_recall_at_k"] + 0.45 * va_m["pos_file_recall_at_k"]
        if sc > best_div_score:
            best_div_score = sc
            best_div_lam = cand_lam

    # Fit per-genome Slot-1 True-Hit Platt `(T, b)` + `conformal_q_hat` independently (`F-06`)
    T_p1, b_p1, q_p1 = calibrate_platt_and_conformal(
        train_records, val_records, w_p1, top_k=3, max_tokens=800, gamma_max=gamma_max
    )
    T_p3, b_p3, q_p3 = calibrate_platt_and_conformal(
        train_records, val_records, w_p3_distilled, top_k=3, max_tokens=800, gamma_max=gamma_max
    )
    T_p2, b_p2, q_p2 = calibrate_platt_and_conformal(
        train_records,
        val_records,
        w_p3_distilled,
        top_k=8,
        max_tokens=4096,
        w_residual=w_p2_res,
        diversity_lambda=best_div_lam,
        gamma_max=gamma_max,
    )

    return {
        "device": dev_str,
        "gpu_name": gpu_name,
        "total_generations": total_generations,
        "pop_size": pop_size,
        "total_genome_evaluations": total_generations * pop_size,
        "evolution_wall_sec": evolve_elapsed_s,
        "gamma_max": gamma_max,
        "phase1_e4b_sniper": {
            "generations": g_p1,
            "w_core": w_p1,
            "train_fitness": tr_p1,
            "val_fitness": va_p1,
            "platt_T": T_p1,
            "platt_b": b_p1,
            "conformal_q_hat": q_p1,
        },
        "phase2_31b_deep": {
            "generations": g_p2,
            "w_anchor": w_p3_distilled,
            "w_residual": w_p2_res,
            "diversity_lambda": best_div_lam,
            "train_fitness": tr_p2,
            "val_fitness": va_p2,
            "platt_T": T_p2,
            "platt_b": b_p2,
            "conformal_q_hat": q_p2,
        },
        "phase3_e4b_sniper_distilled": {
            "generations": g_p3,
            "w_core": w_p3_distilled,
            "train_fitness": tr_p3,
            "val_fitness": va_p3,
            "platt_T": T_p3,
            "platt_b": b_p3,
            "conformal_q_hat": q_p3,
        },
    }


def fit_and_calibrate_tool_gate(when2call_train_path: Path, seed: int = 42) -> Dict[str, Any]:
    """
    Fit & temperature-calibrate the Factorized 2-Stage Causal Tool-Schema Conformal Gate
    (`Stage 1: P(rel | f[0..6])`, `Stage 2: P(slots_ok | f[7..17], rel)`) strictly on
    non-empty-tool rows of `when2call_train_clean.jsonl` (`0` test overlap).
    """
    import torch
    import torch.nn as nn

    rows = [
        json.loads(line)
        for line in when2call_train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    y_map = {"tool_call": 0, "cannot_answer": 1, "request_for_info": 2}
    valid_rows = [r for r in rows if r.get("inferred_label") in y_map and r.get("tools")]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(valid_rows))
    sp = int(0.80 * len(valid_rows))
    tr_rows = [valid_rows[i] for i in perm[:sp]]
    va_rows = [valid_rows[i] for i in perm[sp:]]

    def _build_split_xy(
        split_rows: List[Dict[str, Any]], augment_hard_ca: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_list: List[List[float]] = []
        y_list: List[int] = []
        for r in split_rows:
            lbl = y_map[r["inferred_label"]]
            feats, m_name = lt.extract_tool_call_features(r["question"], r["tools"])
            x_list.append(feats)
            y_list.append(lbl)
            # Augment hard domain-sibling cannot_answer negatives by removing matched target tool
            if augment_hard_ca and lbl == 0 and len(r["tools"]) >= 2 and m_name:
                sib_tools = [
                    t
                    for t in r["tools"]
                    if isinstance(t, Mapping) and str(t.get("function", t).get("name", "")) != m_name
                ]
                if sib_tools:
                    sib_feats, _ = lt.extract_tool_call_features(r["question"], sib_tools)
                    if sib_feats[0] < 0.70:
                        x_list.append(sib_feats)
                        y_list.append(1)
        return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.int64)

    torch.manual_seed(seed)
    X_tr, y_tr = _build_split_xy(tr_rows, augment_hard_ca=True)
    X_va, y_va = _build_split_xy(va_rows, augment_hard_ca=False)

    # Stage 1: Sign-Constrained Monotonic Relevance Head (all 7 features strictly non-decreasing: w1 >= 0, w2 >= 0)
    rel_u1 = nn.Parameter(torch.randn(8, 7) * 0.25 - 0.5)
    rel_b1 = nn.Parameter(torch.zeros(8))
    rel_u2 = nn.Parameter(torch.randn(1, 8) * 0.25 - 0.3)
    rel_b2 = nn.Parameter(torch.tensor([-0.4]))
    opt_rel = torch.optim.AdamW([rel_u1, rel_b1, rel_u2, rel_b2], lr=0.025, weight_decay=3e-3)
    xt_rel = torch.from_numpy(X_tr[:, :7])
    yt_rel = torch.from_numpy((y_tr != 1).astype(np.float32)).unsqueeze(1)
    for _ in range(650):
        opt_rel.zero_grad()
        w1_r = nn.functional.softplus(rel_u1)
        w2_r = nn.functional.softplus(rel_u2)
        h_r = torch.tanh(xt_rel @ w1_r.T + rel_b1)
        lg = h_r @ w2_r.T + rel_b2
        loss = nn.functional.binary_cross_entropy_with_logits(lg, yt_rel)
        loss.backward()
        opt_rel.step()

    # Stage 2: Sign-Constrained Causal Monotonic Slot-Satisfaction Head on relevant rows
    # Causal sign prior on Dims 7..17: [+zero_req, +req_cov, -unfilled, -req_cnt, -miss_num,
    # -miss_id, +ent, +pcov, +cval, -ndef, -opt_only]
    slot_signs = torch.tensor([[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0]], dtype=torch.float32)
    rel_mask_tr = y_tr != 1
    slot_u1 = nn.Parameter(torch.randn(8, 11) * 0.25 - 0.4)
    slot_b1 = nn.Parameter(torch.zeros(8))
    slot_u2 = nn.Parameter(torch.randn(1, 8) * 0.25 - 0.3)
    slot_b2 = nn.Parameter(torch.tensor([0.0]))
    opt_slot = torch.optim.AdamW([slot_u1, slot_b1, slot_u2, slot_b2], lr=0.025, weight_decay=3e-3)
    xt_slot = torch.from_numpy(X_tr[rel_mask_tr, 7:])
    yt_slot = torch.from_numpy((y_tr[rel_mask_tr] == 0).astype(np.float32)).unsqueeze(1)
    for _ in range(700):
        opt_slot.zero_grad()
        w1_s = nn.functional.softplus(slot_u1) * slot_signs
        w2_s = nn.functional.softplus(slot_u2)
        h_s = torch.tanh(xt_slot @ w1_s.T + slot_b1)
        lg_s = h_s @ w2_s.T + slot_b2
        loss_s = nn.functional.binary_cross_entropy_with_logits(lg_s, yt_slot)
        loss_s.backward()
        opt_slot.step()

    with torch.no_grad():
        w1_r_eff = nn.functional.softplus(rel_u1)
        w2_r_eff = nn.functional.softplus(rel_u2)
        w1_s_eff = nn.functional.softplus(slot_u1) * slot_signs
        w2_s_eff = nn.functional.softplus(slot_u2)
        va_z_rel = (
            (torch.tanh(torch.from_numpy(X_va[:, :7]) @ w1_r_eff.T + rel_b1) @ w2_r_eff.T + rel_b2)
            .squeeze(1)
            .numpy()
        )
        va_z_slot = (
            (torch.tanh(torch.from_numpy(X_va[:, 7:]) @ w1_s_eff.T + slot_b1) @ w2_s_eff.T + slot_b2)
            .squeeze(1)
            .numpy()
        )

    y_va_bin = (y_va == 0).astype(np.float64)
    best_t_rel, best_t_slot = 1.0, 1.0
    best_ece = 1e9
    for tr_c in np.linspace(0.75, 2.40, 24):
        p_r = 1.0 / (1.0 + np.exp(-np.clip(va_z_rel / tr_c, -30.0, 30.0)))
        for ts_c in np.linspace(0.75, 2.40, 24):
            p_s = 1.0 / (1.0 + np.exp(-np.clip(va_z_slot / ts_c, -30.0, 30.0)))
            p_tc = p_r * p_s
            brier_c = float(np.mean((p_tc - y_va_bin) ** 2))
            ece_c = compute_ece_10bin(p_tc.tolist(), y_va_bin.tolist()) + 0.15 * brier_c
            if ece_c < best_ece:
                best_ece = ece_c
                best_t_rel, best_t_slot = float(tr_c), float(ts_c)

    p_rel_va = 1.0 / (1.0 + np.exp(-np.clip(va_z_rel / best_t_rel, -30.0, 30.0)))
    p_slot_va = 1.0 / (1.0 + np.exp(-np.clip(va_z_slot / best_t_slot, -30.0, 30.0)))
    p_tool_va = p_rel_va * p_slot_va
    p_cannot_va = 1.0 - p_rel_va
    p_rfi_va = p_rel_va * (1.0 - p_slot_va)

    best_tau = 0.44
    best_bal = -1.0
    for tau_c in np.linspace(0.32, 0.54, 45):
        pred_tc = p_tool_va >= tau_c
        t_rec = float(np.mean(pred_tc[y_va == 0]))
        a_rec = float(np.mean((~pred_tc)[y_va != 0]))
        b_acc = float(np.mean(pred_tc == (y_va == 0)))
        bal = min(t_rec, a_rec) + 0.25 * b_acc
        if bal > best_bal:
            best_bal = bal
            best_tau = float(tau_c)

    pred_final = p_tool_va >= best_tau
    pred_3way = np.where(pred_final, 0, np.where(p_cannot_va >= p_rfi_va, 1, 2))
    val_metrics = {
        "train_rows": int(len(X_tr)),
        "val_rows": int(len(X_va)),
        "val_binary_acc": round(float(np.mean(pred_final == (y_va == 0))), 4),
        "val_tool_call_recall": round(float(np.mean(pred_final[y_va == 0])), 4),
        "val_combined_abstain_recall": round(float(np.mean((~pred_final)[y_va != 0])), 4),
        "val_3way_acc": round(float(np.mean(pred_3way == y_va)), 4),
        "val_top1_ece": compute_ece_10bin(p_tool_va.tolist(), y_va_bin.tolist()),
    }

    return {
        "gate_payload": {
            "schema": "factorized_2stage_v23",
            "tau_tool_call": round(best_tau, 4),
            "temp_rel": round(best_t_rel, 4),
            "temp_slot": round(best_t_slot, 4),
            "rel_w1": [[round(float(v), 5) for v in row] for row in w1_r_eff.detach().cpu().tolist()],
            "rel_b1": [round(float(v), 5) for v in rel_b1.detach().cpu().tolist()],
            "rel_w2": [round(float(v), 5) for v in w2_r_eff.detach().cpu().squeeze(0).tolist()],
            "rel_b2": round(float(rel_b2.detach().cpu().item()), 5),
            "slot_w1": [[round(float(v), 5) for v in row] for row in w1_s_eff.detach().cpu().tolist()],
            "slot_b1": [round(float(v), 5) for v in slot_b1.detach().cpu().tolist()],
            "slot_w2": [round(float(v), 5) for v in w2_s_eff.detach().cpu().squeeze(0).tolist()],
            "slot_b2": round(float(slot_b2.detach().cpu().item()), 5),
        },
        "val_metrics": val_metrics,
    }


def run_clean_subprocess_nano_gate() -> Dict[str, Any]:
    ple_json = DATA_DIR / "ple_dev_29_metadata.json"
    probe_code = (
        "import json, sys, time, psutil; "
        "proc = psutil.Process(); "
        "rss_before = proc.memory_info().rss; "
        f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "import laya_triage as lt; "
        "assert 'torch' not in sys.modules, 'torch imported at startup'; "
        "t0 = time.perf_counter(); "
        f"res = lt.run_triage([{str(ple_json)!r}], mode='shortlist', "
        "query='Clash-free MEP routing through a data hall structural grid', "
        "profile='nano', agent_tier='e4b-sniper', top_k=3); "
        "dt_ms = (time.perf_counter() - t0) * 1000.0; "
        "rss_after = proc.memory_info().rss; "
        "assert 'torch' not in sys.modules, 'torch imported during nano run'; "
        "print(json.dumps({"
        "'torch_in_sys_modules': ('torch' in sys.modules), "
        "'delta_rss_mb': round((rss_after - rss_before) / (1024.0 * 1024.0), 3), "
        "'elapsed_ms': round(dt_ms, 2), "
        "'chunks_evaluated': res['meta']['chunks_evaluated'], "
        "'top1_heading': res['results'][0]['heading'] if res['results'] else ''"
        "}))"
    )
    cp = subprocess.run(
        [sys.executable, "-B", "-c", probe_code],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    out = json.loads(cp.stdout.strip().splitlines()[-1])
    assert out["torch_in_sys_modules"] is False, out
    assert out["delta_rss_mb"] <= 25.0, f"Delta-RSS exceeded 25 MB: {out}"
    assert out["elapsed_ms"] <= 200.0, f"Nano latency exceeded 200 ms: {out}"
    return out


def run_suite1_and_suite3_verification() -> Dict[str, Any]:
    rows_all: List[Dict[str, Any]] = []
    with (DATA_DIR / "routing_labels.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows_all.append(
                {
                    "state": row["state"],
                    "gold": {"team": row["team"], "urgency": int(row["urgency"])},
                }
            )
    fit_rows = rows_all[:40]
    test_rows = rows_all[40:]

    routing_q = {
        "team": {
            "type": "choice",
            "instructions": "Route this incoming message to the owning department.",
            "criteria": {
                "billing": (
                    "Invoices, refunds, GST tax, seat charges, payment cards, "
                    "bank transfers, overdue accounts"
                ),
                "hr": (
                    "Job applications, payslips, leave days, workplace incidents, "
                    "visas, insurance, onboarding, resignation"
                ),
                "pr": (
                    "Press inquiries, journalists, bloggers, CEO quotes, "
                    "conferences, podcasts, media kits, interviews"
                ),
                "sales": (
                    "Product demos, enterprise pricing, NDAs, contracts, "
                    "non-profit discounts, trials, procurement quotes"
                ),
                "technical": (
                    "API 500/502 errors, key rotation, CSV encoding bugs, "
                    "Python SDKs, webhooks, TLS, rate limits"
                ),
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate the operational urgency of this incoming request from 0 (routine) to 2 (critical).",
            "criteria": [
                "0 - routine inquiry, policy question, or non-urgent scheduling request",
                "1 - time-sensitive operational issue, bug, billing discrepancy, or upcoming deadline",
                "2 - critical production outage, blocked account, severe incident, or immediate press/board deadline",
            ],
        },
    }

    zs_pred = lt.predict_direct(states=[r["state"] for r in test_rows], questions=routing_q, model="ensemble")
    zs_hits = sum(
        1 for r, p in zip(test_rows, zs_pred["results"]) if p["answers"]["team"]["choice"] == r["gold"]["team"]
    )
    zs_acc = round(zs_hits / max(1, len(test_rows)), 4)

    with tempfile.TemporaryDirectory(prefix="xlaya_routing_fit_") as tmp_pol:
        fit_meta = lt.fit_policy(
            rows=fit_rows,
            questions=routing_q,
            policy_path=tmp_pol,
            vega_model="0.8b",
            folds=4,
            target_precision=0.95,
        )
        ttt_pred = lt.predict_direct(states=[r["state"] for r in test_rows], questions=routing_q, policy_path=tmp_pol)
        ttt_hits = sum(
            1 for r, p in zip(test_rows, ttt_pred["results"]) if p["answers"]["team"]["choice"] == r["gold"]["team"]
        )
        ttt_acc = round(ttt_hits / max(1, len(test_rows)), 4)

    boundary_items = json.loads((DATA_DIR / "xerj_1108_boundary.json").read_text(encoding="utf-8"))
    b_states = [f"Subject: {it['subject']}\n{it['text']}" for it in boundary_items]
    b_gold = [it["label"] for it in boundary_items]

    by_lbl: Dict[str, List[int]] = {}
    for idx_b, lbl in enumerate(b_gold):
        by_lbl.setdefault(lbl, []).append(idx_b)
    b_fit_idxs: List[int] = []
    b_eval_idxs: List[int] = []
    for _, idxs_l in sorted(by_lbl.items()):
        cut = int(round(len(idxs_l) * 0.60))
        b_fit_idxs.extend(idxs_l[:cut])
        b_eval_idxs.extend(idxs_l[cut:])

    boundary_q = {
        "watcher": {
            "type": "choice",
            "instructions": (
                "Classify this email strictly by true sender obligation: "
                "'personal' (friend casual catch-up or splitting personal trip/dinner expenses), "
                "'newsletter' (broadcast digest, company roundup, or marketing poll asking readers to hit reply), "
                "'needs-reply' (direct operational blocker or work deliverable requiring action), "
                "or 'invoice' (commercial vendor invoice, billing statement, or accounts-payable charge)."
            ),
            "criteria": {
                "personal": "Personal message from a friend about hanging out, dinner, or splitting trip/ticket costs",
                "newsletter": "Mass weekly digest, product release roundup, or newsletter engagement question",
                "needs-reply": (
                    "Direct 1-to-1 work or operational blocker waiting on a concrete deliverable or sign-off"
                ),
                "invoice": "Commercial vendor invoice, billing receipt, PO payment due notice, or accounting statement",
            },
        }
    }

    with tempfile.TemporaryDirectory(prefix="xlaya_xerj_fit_") as tmp_xpol:
        lt.fit_policy(
            rows=[{"state": b_states[i], "gold": {"watcher": b_gold[i]}} for i in b_fit_idxs],
            questions=boundary_q,
            policy_path=tmp_xpol,
            vega_model="0.8b",
            folds=4,
            target_precision=0.98,
        )
        b_eval_preds = lt.predict_direct(
            states=[b_states[i] for i in b_eval_idxs],
            questions=boundary_q,
            policy_path=tmp_xpol,
            views=2,
        )

    b_correct = 0
    b_wrong_confident_80 = 0
    b_auto_total = 0
    b_auto_correct = 0
    for idx_orig, pred_item in zip(b_eval_idxs, b_eval_preds["results"]):
        ans_w = pred_item["answers"]["watcher"]
        pred_lbl = ans_w["choice"]
        conf = float(ans_w.get("confidence", 0.0))
        abstain = bool(ans_w.get("abstain", False))
        gold_lbl = b_gold[idx_orig]
        if pred_lbl == gold_lbl:
            b_correct += 1
        elif conf >= 0.80 and not abstain:
            b_wrong_confident_80 += 1
        if not abstain:
            b_auto_total += 1
            if pred_lbl == gold_lbl:
                b_auto_correct += 1

    ple_data = json.loads((DATA_DIR / "ple_dev_29_metadata.json").read_text(encoding="utf-8"))
    ple_tasks = ple_data["tasks"]
    ple_q = {
        "grading_mode": {
            "type": "choice",
            "instructions": (
                "Predict whether this engineering task uses deterministic software/CAD "
                "checks or an open-ended rubric memo."
            ),
            "criteria": {
                "deterministic_checks": (
                    "Produces an executable script (.py), IFC BIM model (.ifc), or CAD "
                    "package graded by automated checks"
                ),
                "rubric_memo": (
                    "Validation memo, resubmittal sheet review, or bond estimate review "
                    "with final_message narrative"
                ),
            },
        }
    }
    ple_states = [
        (
            f"Title: {t['title']} | Domain: {t['domain']} | "
            f"Deliverables: {', '.join(t['deliverables'])} | FinalMessage: {t['final_message']}"
        )
        for t in ple_tasks
    ]
    ple_nano_pred = lt.predict_direct(states=ple_states, questions=ple_q, profile="nano")
    ple_ens_pred = lt.predict_direct(states=ple_states, questions=ple_q, model="ensemble")

    def _ple_gold_bucket(t_item: Mapping[str, Any]) -> str:
        return "rubric_memo" if t_item["grading"] == "rubric" else "deterministic_checks"

    ple_nano_hits = sum(
        1
        for t_item, p_item in zip(ple_tasks, ple_nano_pred["results"])
        if p_item["answers"]["grading_mode"]["choice"] == _ple_gold_bucket(t_item)
    )
    ple_ens_hits = sum(
        1
        for t_item, p_item in zip(ple_tasks, ple_ens_pred["results"])
        if p_item["answers"]["grading_mode"]["choice"] == _ple_gold_bucket(t_item)
    )

    return {
        "suite1_routing_labels": {
            "fit_count": len(fit_rows),
            "test_count": len(test_rows),
            "stage1_ensemble_zero_shot_acc": zs_acc,
            "stage2_ttthead_fitted_acc": ttt_acc,
            "calibration_saved": bool(fit_meta.get("ok")),
        },
        "suite1_xerj_1108_boundary": {
            "total_items": len(boundary_items),
            "fit_count": len(b_fit_idxs),
            "blind_eval_count": len(b_eval_idxs),
            "accuracy": round(b_correct / max(1, len(b_eval_idxs)), 4),
            "auto_route_precision": round(b_auto_correct / max(1, b_auto_total), 4),
            "auto_route_coverage": round(b_auto_total / max(1, len(b_eval_idxs)), 4),
            "wrong_confident_p80_rate": round(b_wrong_confident_80 / max(1, len(b_eval_idxs)), 4),
        },
        "suite3_ple_29_cross_domain": {
            "task_count": len(ple_tasks),
            "nano_zero_shot_acc": round(ple_nano_hits / len(ple_tasks), 4),
            "ensemble_zero_shot_acc": round(ple_ens_hits / len(ple_tasks), 4),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="XLaya v2.3.0 Repo-Disjoint 3-Phase CUDA Sieve Evolution & Calibration"
    )
    parser.add_argument(
        "--manifest",
        default=os.environ.get("XLAYA_DISJOINT_MANIFEST", ""),
        help="Path to disjoint_swe_v23_manifest.json (`0` overlap with Kaggle 129 & SWE-bench_Verified)",
    )
    parser.add_argument(
        "--when2call-train",
        default=os.environ.get("XLAYA_WHEN2CALL_TRAIN", ""),
        help="Path to clean When2Call train JSONL (`0` overlap with When2Call test)",
    )
    parser.add_argument(
        "--verify-zero-leakage",
        action="store_true",
        help="Verify 0% repo/commit/task leakage in --manifest and exit",
    )
    parser.add_argument(
        "--skip-suite13",
        action="store_true",
        help="Skip neural Suite 1 & Suite 3 regression run when running standalone sieve evolution",
    )
    parser.add_argument(
        "--recalibrate-existing",
        action="store_true",
        help="Recalibrate Slot-1 Platt (T, b) and conformal_q_hat on existing genomes without re-running CUDA ES",
    )
    parser.add_argument(
        "--recalibrate-tool-gate-only",
        action="store_true",
        help="Refit & calibrate only tool_gate on --when2call-train while preserving SWE sieve genomes",
    )
    parser.add_argument("--generations", type=int, default=5000, help="Total evolutionary generations (default: 5000)")
    parser.add_argument("--pop-size", type=int, default=256, help="Population size per generation (default: 256)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed (default: 42)")
    args = parser.parse_args(argv)

    if not args.manifest:
        raise SystemExit("Error: --manifest (or XLAYA_DISJOINT_MANIFEST) is required for zero-leakage v2.3 evolution.")
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"Error: manifest file not found: {manifest_path}")

    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    leak_report = verify_manifest_zero_leakage(manifest_obj)
    if args.verify_zero_leakage:
        print(json.dumps({"ok": True, "version": lt.__version__, **leak_report}, indent=2))
        return 0

    if args.recalibrate_tool_gate_only:
        if not args.when2call_train or not Path(args.when2call_train).is_file():
            raise SystemExit("Error: --when2call-train is required with --recalibrate-tool-gate-only")
        tg_res = fit_and_calibrate_tool_gate(Path(args.when2call_train), seed=args.seed)
        genomes_path = POLICIES_DIR / "sieve_genomes.json"
        g_obj = json.loads(genomes_path.read_text(encoding="utf-8"))
        g_obj["tool_gate"] = tg_res["gate_payload"]
        g_txt = json.dumps(g_obj, indent=2) + "\n"
        assert len(g_txt.encode("utf-8")) < 15360, "sieve_genomes.json must stay < 15 KB"
        genomes_path.write_text(g_txt, encoding="utf-8")
        (SKILL_RES_DIR / "sieve_genomes.json").write_text(g_txt, encoding="utf-8")
        out_json = BENCH_DIR / "xlaya_v23_evolution_results.json"
        if out_json.is_file():
            r_obj = json.loads(out_json.read_text(encoding="utf-8"))
            r_obj["when2call_tool_gate_val"] = tg_res["val_metrics"]
            r_obj["funnel2_gates"]["sieve_genomes_bytes"] = len(g_txt.encode("utf-8"))
            out_json.write_text(json.dumps(r_obj, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "ok": True,
                    "when2call_tool_gate_val": tg_res["val_metrics"],
                    "sieve_genomes_bytes": len(g_txt.encode("utf-8")),
                },
                indent=2,
            )
        )
        return 0

    dataset = build_disjoint_swe_feature_dataset(manifest_path)
    records = dataset["records"]
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]

    existing_res_path = BENCH_DIR / "xlaya_v23_evolution_results.json"
    if args.recalibrate_existing and existing_res_path.is_file():
        prev_res = json.loads(existing_res_path.read_text(encoding="utf-8"))
        evo = prev_res["funnel1_cuda_evolution"]
        w_p1 = evo["phase1_e4b_sniper"]["w_core"]
        w_p3 = evo["phase3_e4b_sniper_distilled"]["w_core"]
        w_p2_res = evo["phase2_31b_deep"]["w_residual"]
        div_lam = float(evo["phase2_31b_deep"]["diversity_lambda"])
        T_p1, b_p1, q_p1 = calibrate_platt_and_conformal(
            train_recs, val_recs, w_p1, top_k=3, max_tokens=800, gamma_max=1.45
        )
        T_p3, b_p3, q_p3 = calibrate_platt_and_conformal(
            train_recs, val_recs, w_p3, top_k=3, max_tokens=800, gamma_max=1.45
        )
        T_p2, b_p2, q_p2 = calibrate_platt_and_conformal(
            train_recs,
            val_recs,
            w_p3,
            top_k=8,
            max_tokens=4096,
            w_residual=w_p2_res,
            diversity_lambda=div_lam,
            gamma_max=1.45,
        )
        evo["phase1_e4b_sniper"].update({"platt_T": T_p1, "platt_b": b_p1, "conformal_q_hat": q_p1})
        evo["phase3_e4b_sniper_distilled"].update({"platt_T": T_p3, "platt_b": b_p3, "conformal_q_hat": q_p3})
        evo["phase2_31b_deep"].update({"platt_T": T_p2, "platt_b": b_p2, "conformal_q_hat": q_p2})
    else:
        evo = run_three_phase_cuda_evolution(
            train_recs,
            val_recs,
            total_generations=args.generations,
            pop_size=args.pop_size,
            seed=args.seed,
            gamma_max=1.45,
        )

    p1 = evo["phase1_e4b_sniper"]
    p2 = evo["phase2_31b_deep"]
    p3 = evo["phase3_e4b_sniper_distilled"]

    # Verify non-degenerate learned weights (`G-01`, `G-03`, `G-08`)
    assert p3["w_core"][5] > 0.03 and p3["w_core"][7] > 0.03, p3["w_core"]
    assert p2["w_residual"][0] >= 0.05 and p2["w_residual"][5] >= 0.06, p2["w_residual"]

    tool_gate_info: Dict[str, Any] = {}
    if args.when2call_train and Path(args.when2call_train).is_file():
        tg_res = fit_and_calibrate_tool_gate(Path(args.when2call_train), seed=args.seed)
        tool_gate_payload = tg_res["gate_payload"]
        tool_gate_info = tg_res["val_metrics"]
    else:
        tool_gate_payload = dict(lt.DEFAULT_SIEVE_GENOMES["tool_gate"])

    genomes_payload: Dict[str, Any] = {
        "schema_version": "2.3.0",
        "license": "Apache-2.0",
        "feature_names": list(lt.SIEVE_FEATURE_NAMES),
        "gamma_max": 1.45,
        "evolution_meta": {
            "device": evo["device"],
            "gpu_name": evo["gpu_name"],
            "training_regime": "repo_disjoint_base_commit_swe_lite_dev_plus_loc_bench_v1",
            "zero_leakage_verified": True,
            "train_tasks": len(train_recs),
            "val_tasks": len(val_recs),
            "total_generations": evo["total_generations"],
            "population_size": evo["pop_size"],
            "total_genome_evaluations": evo["total_genome_evaluations"],
            "evolution_wall_sec": evo["evolution_wall_sec"],
            "seed": args.seed,
        },
        "e4b_sniper_p1": {
            "w_core": p1["w_core"],
            "w_halo": p1["w_core"],
            "top_k": 3,
            "token_budget": 800,
            "lambda_file_div": 0.35,
            "platt_T": p1["platt_T"],
            "platt_b": p1["platt_b"],
            "conformal_q_hat": p1["conformal_q_hat"],
        },
        "e4b_sniper_distilled": {
            "w_core": p3["w_core"],
            "w_halo": p3["w_core"],
            "top_k": 3,
            "token_budget": 800,
            "lambda_file_div": 0.35,
            "platt_T": p3["platt_T"],
            "platt_b": p3["platt_b"],
            "conformal_q_hat": p3["conformal_q_hat"],
        },
        "31b_deep": {
            "w_core": p2["w_anchor"],
            "w_halo": p2["w_residual"],
            "anchor_slots": 3,
            "top_k": 8,
            "token_budget": 4096,
            "lambda_file_div": p2["diversity_lambda"],
            "platt_T": p2["platt_T"],
            "platt_b": p2["platt_b"],
            "conformal_q_hat": p2["conformal_q_hat"],
        },
        "tool_gate": tool_gate_payload,
    }

    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    SKILL_RES_DIR.mkdir(parents=True, exist_ok=True)
    genomes_path = POLICIES_DIR / "sieve_genomes.json"
    genomes_text = json.dumps(genomes_payload, indent=2) + "\n"
    assert len(genomes_text.encode("utf-8")) < 15360, "sieve_genomes.json must stay < 15 KB"
    genomes_path.write_text(genomes_text, encoding="utf-8")
    (SKILL_RES_DIR / "sieve_genomes.json").write_text(genomes_text, encoding="utf-8")

    def _strip_files(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if k != "per_task_selected_files"}

    w_bm25 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    val_scorecard = {
        "bm25_only_k3_800tok": _strip_files(
            evaluate_genome_on_records(
                val_recs,
                w_bm25,
                top_k=3,
                max_tokens=800,
                platt_T=p3["platt_T"],
                platt_b=p3["platt_b"],
                conformal_q_hat=p3["conformal_q_hat"],
            )
        ),
        "bm25_only_k8_4096tok": _strip_files(
            evaluate_genome_on_records(
                val_recs,
                w_bm25,
                top_k=8,
                max_tokens=4096,
                platt_T=p2["platt_T"],
                platt_b=p2["platt_b"],
                conformal_q_hat=p2["conformal_q_hat"],
            )
        ),
        "phase1_e4b_sniper_k3_800tok": _strip_files(
            evaluate_genome_on_records(
                val_recs,
                p1["w_core"],
                top_k=3,
                max_tokens=800,
                platt_T=p1["platt_T"],
                platt_b=p1["platt_b"],
                conformal_q_hat=p1["conformal_q_hat"],
            )
        ),
        "phase2_31b_deep_k8_4096tok": _strip_files(
            evaluate_genome_on_records(
                val_recs,
                p2["w_anchor"],
                top_k=8,
                max_tokens=4096,
                w_residual=p2["w_residual"],
                anchor_slots=3,
                diversity_lambda=p2["diversity_lambda"],
                platt_T=p2["platt_T"],
                platt_b=p2["platt_b"],
                conformal_q_hat=p2["conformal_q_hat"],
            )
        ),
        "phase3_e4b_sniper_distilled_k3_800tok": _strip_files(
            evaluate_genome_on_records(
                val_recs,
                p3["w_core"],
                top_k=3,
                max_tokens=800,
                platt_T=p3["platt_T"],
                platt_b=p3["platt_b"],
                conformal_q_hat=p3["conformal_q_hat"],
            )
        ),
    }

    assert val_scorecard["phase3_e4b_sniper_distilled_k3_800tok"]["neg_wrong_confident_80_rate"] == 0.0
    assert val_scorecard["phase2_31b_deep_k8_4096tok"]["neg_wrong_confident_80_rate"] == 0.0

    nano_gate = run_clean_subprocess_nano_gate()
    cross_suites = {} if args.skip_suite13 else run_suite1_and_suite3_verification()
    if cross_suites:
        assert cross_suites["suite1_xerj_1108_boundary"]["wrong_confident_p80_rate"] == 0.0, cross_suites

    results_summary = {
        "ok": True,
        "version": lt.__version__,
        "leak_audit": leak_report,
        "dataset": {
            "snapshot_count": dataset["snapshot_count"],
            "total_py_files": dataset["total_py_files"],
            "total_chunks_evaluated": dataset["total_chunks_evaluated"],
            "mean_chunks_per_task": dataset["mean_chunks_per_task"],
            "splits": {"train": len(train_recs), "val": len(val_recs)},
            "feature_extract_sec": dataset["feature_extract_sec"],
        },
        "funnel1_cuda_evolution": evo,
        "disjoint_val_scorecard": val_scorecard,
        "when2call_tool_gate_val": tool_gate_info,
        "funnel2_gates": {
            "clean_subprocess_nano": nano_gate,
            "sieve_genomes_bytes": len(genomes_text.encode("utf-8")),
        },
        "cross_benchmarks": cross_suites,
    }

    out_json = BENCH_DIR / "xlaya_v23_evolution_results.json"
    out_json.write_text(json.dumps(results_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
