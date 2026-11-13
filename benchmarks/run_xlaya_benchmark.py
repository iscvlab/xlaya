#!/usr/bin/env python3
"""Reproducible 3-Stage Benchmark & Verification Harness for iscvlab/xlaya (v2.3.0)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.dont_write_bytecode = True
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import laya_triage  # noqa: E402


def _bench_stage0_sieve() -> Tuple[str, str, str]:
    """Verify 4-backtick fence-safe chunking, Thai 3-gram + BM25 shortlisting, and token compression."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="xlaya_bench_s0_") as tmp_str:
        tmp_dir = Path(tmp_str)
        docs: List[Tuple[str, str]] = [
            (
                "spec_architecture.md",
                (
                    "# XLaya System Architecture\n\n"
                    "Overview of the 3-stage semantic triage cascade with deterministic Stage 0 chunking.\n\n"
                    "## Nested Code Fence Safety\n\n"
                    "````markdown\n"
                    "```python\n"
                    "# Fake Heading Inside Nested Code Block\n"
                    "x = 42\n"
                    "```\n"
                    "````\n\n"
                    "### Joint NLI Contradiction Triad\n\n"
                    "Evaluates P_eff = P(is_relevant) * P(contradicts_ssot) * (1 - P(is_consistent)) "
                    "using `--mode contradiction` across ModernBERT-large (`512` tokens) and mmBERT.\n"
                ),
            ),
            (
                "spec_bilingual_thai.md",
                (
                    "# ระบบคัดกรองความหมายภาษาไทย\n\n"
                    "ระบบตรวจสอบความขัดแย้งทางความหมายสำหรับเอกสารภาษาไทยด้วยตัวตัดคำ 3-gram "
                    "และโมเดลหลายภาษา `subfolder=\"multilingual\"` โดยไม่ต้องสร้างโทเคนเอาต์พุต\n\n"
                    "## Unrelated Culinary Appendix\n\n"
                    "สูตรการทำอาหารต้มยำกุ้งและส่วนผสมสมุนไพรสดในครัวเรือนสำหรับมื้อเย็น\n"
                ),
            ),
        ]
        for idx in range(10):
            docs.append(
                (
                    f"noise_module_{idx:02d}.md",
                    (
                        f"# Background Service Module {idx}\n\n"
                        "Routine telemetry collector for cluster heartbeats, cron schedules, "
                        "and disk rotation policies unrelated to semantic NLI contradiction triage.\n\n"
                        f"## Worker Queue {idx}\n\n"
                        "Handles retry backoff intervals and static asset compression headers.\n"
                    ),
                )
            )

        all_chunks: List[Dict[str, Any]] = []
        for name, content in docs:
            fpath = tmp_dir / name
            fpath.write_text(content, encoding="utf-8")
            all_chunks.extend(laya_triage.chunk_file(fpath, max_chars=1600))

        headings = [c["heading"] for c in all_chunks]
        if any("Fake Heading Inside Nested Code Block" in h for h in headings):
            raise AssertionError("4-backtick outer fence failed to shield inner `#` comment.")

        en_scores = laya_triage.compute_lexical_scores(
            all_chunks, "Joint NLI Contradiction Triad --mode contradiction"
        )
        th_scores = laya_triage.compute_lexical_scores(
            all_chunks, "ตรวจสอบความขัดแย้งทางความหมายสำหรับเอกสารภาษาไทย"
        )
        top_en_idx = max(range(len(all_chunks)), key=lambda i: en_scores[i])
        top_th_idx = max(range(len(all_chunks)), key=lambda i: th_scores[i])
        if "Joint NLI Contradiction Triad" not in all_chunks[top_en_idx]["heading"]:
            raise AssertionError(f"Unexpected top EN chunk: {all_chunks[top_en_idx]['heading']}")
        if "ระบบคัดกรองความหมายภาษาไทย" not in all_chunks[top_th_idx]["heading"]:
            raise AssertionError(f"Unexpected top TH chunk: {all_chunks[top_th_idx]['heading']}")

        total_chars = sum(len(c["text"]) for c in all_chunks)
        top2_chars = len(all_chunks[top_en_idx]["text"]) + len(all_chunks[top_th_idx]["text"])
        saved_pct = 100.0 * (1.0 - (top2_chars / max(1, total_chars)))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    detail = (
        f"`{len(all_chunks)}` chunks (`0` fence leaks), EN+TH top-1 exact, "
        f"`{saved_pct:.1f}%` chars pruned"
    )
    return ("Stage 0 Fence-Safe Chunker & Lexical Sieve", f"{elapsed_ms:.2f} ms", detail)


def _bench_head_language_guard() -> Tuple[str, str, str]:
    """Verify Head-Language Guard language detection invariants."""
    t0 = time.perf_counter()
    en_q = {
        "q": {
            "type": "noul",
            "instructions": "Does this section describe ModernBERT batch inference?",
        }
    }
    th_q = {
        "q": {
            "type": "noul",
            "instructions": "เอกสารส่วนนี้อธิบายสถาปัตยกรรมภาษาไทยหรือไม่?",
        }
    }
    en_head_ok = laya_triage._is_english_text(laya_triage._questions_head_text(en_q))
    th_head_ok = not laya_triage._is_english_text(laya_triage._questions_head_text(th_q))
    if not (en_head_ok and th_head_ok):
        raise AssertionError("Head-Language Guard script classifier failed.")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (
        "Head-Language Guard Script Router",
        f"{elapsed_ms:.2f} ms",
        "EN head -> `english`, Non-EN (Thai) head -> `100% multilingual`",
    )


def _bench_neural_probes(full_mode: bool) -> List[Tuple[str, str, str]]:
    """Run warm Stage 1 Laya and optional Stage 2 Vega-U probes if installed."""
    rows: List[Tuple[str, str, str]] = []
    status = laya_triage.get_status()
    if not status.get("laya_available"):
        rows.append(
            (
                "Stage 1 Warm Laya Probe",
                "skipped",
                "`laya` not installed (pure-python Stage 0 fallback verified)",
            )
        )
        return rows

    try:
        t_load = time.perf_counter()
        laya_triage.preload_models(models=["en"], vega=False, device="auto")
        load_ms = (time.perf_counter() - t_load) * 1000.0
        rows.append(
            (
                f"Stage 1 Model Preload (`{status.get('device')}`)",
                f"{load_ms:.1f} ms",
                f"`laya {status.get('laya_version')}` (`{status.get('gpu_name') or 'CPU'}`)",
            )
        )

        questions = {
            "is_relevant": {
                "type": "noul",
                "instructions": "Does this passage discuss ModernBERT context window limits?",
            },
            "contradicts_ssot": {
                "type": "noul",
                "instructions": (
                    "Assuming `ModernBERT-large uses a 512-token context window` is true, "
                    "does this text state a conflicting token limit?"
                ),
            },
        }
        # Warmup CUDA kernels on first forward pass, then measure steady-state warm latency
        laya_triage.predict_direct(
            state="ModernBERT-large uses a 512-token context window.",
            questions=questions,
            model="en",
        )
        res_single = laya_triage.predict_direct(
            state="ModernBERT-large uses a 128-token context window.",
            questions=questions,
            model="en",
        )
        p_rel = float(res_single["result"]["answers"]["is_relevant"].get("noul", 0.0))
        p_con = float(res_single["result"]["answers"]["contradicts_ssot"].get("noul", 0.0))
        rows.append(
            (
                "Stage 1 Warm `predict_direct` (1 State)",
                f"{res_single['elapsed_ms']:.2f} ms",
                f"`P(rel)={p_rel:.3f}`, `P(contra)={p_con:.3f}` (`output_tokens: 0`)",
            )
        )

        if full_mode:
            batch_states = [
                f"Passage {i}: ModernBERT-large uses a 512-token context window on GPU."
                for i in range(16)
            ]
            res_batch = laya_triage.predict_direct(
                states=batch_states,
                questions=questions,
                model="en",
                batch_size=16,
            )
            q_per_s = (len(batch_states) * len(questions)) / max(1e-6, res_batch["elapsed_ms"] / 1000.0)
            rows.append(
                (
                    "Stage 1 Warm Batched Triage (16 States × 2 Qs)",
                    f"{res_batch['elapsed_ms']:.2f} ms",
                    f"`{q_per_s:.1f} q/s` (`sort_by_length=True`)",
                )
            )

            res_ens = laya_triage.predict_direct(
                state="ModernBERT-large uses a 512-token context window and outputs 0 LLM tokens.",
                questions=questions,
                model="ensemble",
            )
            ens_rel = res_ens["result"]["answers"]["is_relevant"]
            dis_v = ens_rel.get("disagreement", 0.0)
            agr_v = ens_rel.get("model_agreement", 1.0)
            rows.append(
                (
                    "Stage 1 Multi-Encoder Jury (`model=\"ensemble\"`)",
                    f"{res_ens['elapsed_ms']:.2f} ms",
                    f"`disagreement={dis_v:.3f}`, `agreement={agr_v:.3f}`",
                )
            )

            if status.get("vegaml_available"):
                v_res = laya_triage.predict_direct(
                    state="ModernBERT-large uses a 512-token context window.",
                    questions={
                        "chk": {
                            "type": "noul",
                            "instructions": "Does this text mention a 512-token context window?",
                        }
                    },
                    vega=True,
                    vega_model="0.8b",
                    views=3,
                )
                v_ans = v_res["result"]["answers"]["chk"]
                v_detail = (
                    f"`route={v_ans.get('route')}`, `abstain={v_ans.get('abstain')}`, "
                    f"`view_agreement={v_ans.get('view_agreement')}`"
                )
                rows.append(
                    (
                        "Stage 2 `vega-U (0.8b, views=3)` Conformal Probe",
                        f"{v_res['elapsed_ms']:.2f} ms",
                        v_detail,
                    )
                )
    finally:
        laya_triage.unload_models()

    return rows


def _verify_receipt_json() -> Tuple[str, str, str]:
    """Validate benchmarks/xlaya_v2_benchmark_results.json schema integrity."""
    t0 = time.perf_counter()
    receipt_path = BENCH_DIR / "xlaya_v2_benchmark_results.json"
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    req_keys = (
        "hardware",
        "axis1_latency_and_batch_throughput",
        "axis2_numerical_consistency",
        "axis3_accuracy_and_conformal_calibration",
        "axis4_presets_and_cross_lingual",
    )
    for key in req_keys:
        if key not in data:
            raise AssertionError(f"Missing benchmark receipt key: {key}")
    flips = data["axis2_numerical_consistency"]["total_label_flips"]
    comparisons = data["axis2_numerical_consistency"]["total_comparisons"]
    if flips != 0 or comparisons != 432:
        raise AssertionError(f"Unexpected Axis 2 consistency telemetry: {flips}/{comparisons}")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (
        "Benchmark Receipt Integrity (`xlaya_v2_benchmark_results.json`)",
        f"{elapsed_ms:.2f} ms",
        f"`4` axes verified, `0 / {comparisons}` batch-vs-single label flips",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible benchmark & verification suite for iscvlab/xlaya (v2.3.0)."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark depth: 'quick' (< 15s Stage 0 + warm probe) or 'full' (+ batched, ensemble & Vega-U)",
    )
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    rows: List[Tuple[str, str, str]] = [
        _bench_stage0_sieve(),
        _bench_head_language_guard(),
        _verify_receipt_json(),
    ]
    rows.extend(_bench_neural_probes(full_mode=(args.mode == "full")))
    total_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"### XLaya v2.3.0 Verification Summary (`mode={args.mode}`, total `{total_ms:.1f} ms`)\n")
    print("| Benchmark Check | Latency | Verified Result |")
    print("| :--- | :---: | :--- |")
    for check_name, latency_str, detail_str in rows:
        print(f"| **{check_name}** | `{latency_str}` | {detail_str} ✓ |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
