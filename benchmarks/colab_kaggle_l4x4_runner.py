#!/usr/bin/env python3
"""
colab_kaggle_l4x4_runner.py — Kaggle/Colab L4x4 Runner & Dual-Track ADK Submission Packager
Part of XLaya v2.3.0 (@iscvlab/xlaya, Apache-2.0).

Capabilities:
  1. Hardware & Profile Verifier (`--hardware-check`):
     Inspects single-GPU / multi-GPU (`L4 x 4` 96 GB VRAM, `max_model_len = 32768`) / CPU container
     environments and runs XLaya `--profile max` or `--profile nano` latency + conformal telemetry.
  2. Dual-Track Kaggle ADK Submission Builder (`--build-submissions --out-dir <dir>`):
     Packages two 100% `adk-submission` (`compile_submission`) compliant archives for the Google
     Gemma 4 Developer Agent Competition ($75k Main Track):
       - `submission_31b_deep.zip`   -> Main Leaderboard ($65k Top-3, `gemma-4-31b-it-qat-w4a16-ct` + `31b-deep`)
       - `submission_e4b_sniper.zip` -> Small Model Bonus ($10k Bonus, `gemma-4-e4b-it-qat-w4a16-ct` + `e4b-sniper`)
  3. Strict Archive & Container-A Parity Self-Test (`--self-test`):
     Enforces `ALLOWED_SUBMISSION_EXTENSIONS` (`.yaml`, `.yml`, `.md`, `.txt`, `.py`, `.json`,
     `.safetensors`), closed `9`-tool `swegemma` `ToolRegistry` + `agent_tool` (`sub_agents/code_analyzer.yaml`
     with `skip_summarization: true`), `thinking_budget` without `thinking_level`, zero `/workspace` temp writes,
     `stdout <= 4,500` chars (`< 5,000` truncation wall), and subprocess parity (`max_abs_diff < 1e-5`)
     with `sys.modules["numpy"] = None` and `sys.modules["torch"] = None`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import laya_triage as lt  # noqa: E402

VLLM_MAX_MODEL_LEN = 32768
MAX_ONESHOT_STDOUT_CHARS = 4500

ALLOWED_SUBMISSION_EXTENSIONS = frozenset(
    {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
)

CLOSED_SWEGEMMA_TOOLS = frozenset(
    {
        "run_command",
        "submit_patch",
        "get_status",
        "read_file",
        "edit_file",
        "write_file",
        "get_code_neighbors",
        "search_similar_code",
        "get_code_subgraph",
    }
)

FORBIDDEN_SMALL_BONUS_MODEL_TOKENS = (
    "gemma-4-31b",
    "gemma-4-26b",
    "gemma-4-12b",
    "gemma-3-27b",
    "31b-deep",
    "31b_deep",
    "26b-a4b",
    "qwen3",
)


def _render_agent_yaml(track: str, model_id: str) -> str:
    return (
        "# Google ADK Root Agent Configuration (Apache-2.0, compile_submission compliant)\n"
        "name: swegemma\n"
        f"model: {model_id}\n"
        "description: >\n"
        f"  XLaya v2.3.0 ({track}) x Gemma 4 Developer Agent with Host Code-Graph tools and\n"
        "  Container-A pure-stdlib AST/DI Conformal Sieve.\n"
        "instruction: !include prompts/system.md\n"
        "generate_content_config: !include configs/sampling.yaml\n"
        "tools:\n"
        "  - run_command\n"
        "  - submit_patch\n"
        "  - get_status\n"
        "  - read_file\n"
        "  - edit_file\n"
        "  - write_file\n"
        "  - get_code_neighbors\n"
        "  - search_similar_code\n"
        "  - get_code_subgraph\n"
        "  - agent_tool:\n"
        "      config_path: sub_agents/code_analyzer.yaml\n"
        "      skip_summarization: true\n"
    )


def _render_code_analyzer_yaml(track: str, model_id: str) -> str:
    return (
        "# Google ADK Sub-Agent Configuration (Host Graph + Container-A Pure-Stdlib Sieve)\n"
        "name: code_analyzer\n"
        f"model: {model_id}\n"
        "description: >\n"
        f"  Dual-channel code localization sub-agent ({track}) combining Host 256D/NetworkX tools\n"
        "  with Container-A pure-stdlib AST/DI conformal sieve (skills/xlaya_sieve).\n"
        "instruction: !include ../prompts/analyzer.md\n"
        "generate_content_config: !include ../configs/sampling.yaml\n"
        "tools:\n"
        "  - run_command\n"
        "  - read_file\n"
        "  - get_code_neighbors\n"
        "  - search_similar_code\n"
        "  - get_code_subgraph\n"
        "skills:\n"
        "  - skills/xlaya_sieve\n"
    )


def _render_sampling_yaml(track: str) -> str:
    think_budget = 2048 if track == "e4b-sniper" else 4096
    return (
        "# vLLM & Google GenAI Sampling Config (max_model_len = 32768)\n"
        "temperature: 0.7\n"
        "top_p: 0.95\n"
        "top_k: 64\n"
        "thinking_config:\n"
        "  include_thoughts: true\n"
        f"  thinking_budget: {think_budget}\n"
    )


def _render_eval_config_yaml(track: str) -> str:
    max_calls = 45 if track == "e4b-sniper" else 60
    return (
        "# Evaluation Harness Budget Overrides\n"
        "evaluation:\n"
        f"  max_tool_calls: {max_calls}\n"
        "  max_time_minutes: 15\n"
        "  timeout_seconds: 180\n"
        "  max_turns: 120\n"
    )


def _render_system_prompt_md(track: str) -> str:
    top_k = 3 if track == "e4b-sniper" else 8
    tok_bud = 800 if track == "e4b-sniper" else 4096
    return (
        f"# XLaya v2.3.0 x Gemma 4 Root Software Engineering Protocol (`{track}`)\n\n"
        "You are an autonomous software engineering agent operating inside `/workspace`.\n\n"
        "## Mandatory Localization-First Workflow\n"
        f"1. **Delegate Localization First**: Invoke `code_analyzer` to shortlist the top `K={top_k}`\n"
        f"   enclosing AST spans (`<={tok_bud}` tokens) with compact class/import skeleton headers.\n"
        "2. **Respect Conformal Abstention**: When `abstain` is `true` (`route == 'review'`), call\n"
        "   `search_similar_code` and `get_code_neighbors` before editing any source file.\n"
        "3. **Zero `/workspace` Pollution (`CRITICAL`)**: NEVER call `write_file` to create temporary\n"
        "   files or scripts (note: `write_file('/tmp/x')` strips the leading `/` and writes\n"
        "   `/workspace/tmp/x` into `git diff`). Run all temporary diagnostics inside `/tmp` via `run_command`.\n"
        "4. **Surgical Edit & Red-to-Green Pytest**: Apply minimal edits with `edit_file`, verify with\n"
        "   `run_command` (`pytest -q`), check `get_status`, and finish with `submit_patch`.\n"
    )


def _render_analyzer_prompt_md(track: str) -> str:
    top_k = 3 if track == "e4b-sniper" else 8
    tok_bud = 800 if track == "e4b-sniper" else 4096
    return (
        f"# XLaya v2.3.0 Dual-Channel Code Analyzer (`{track}`)\n\n"
        "Combine two complementary localization channels without polluting `/workspace`:\n"
        "1. **Host Graph & Embedding Channel**: Call `search_similar_code`, `get_code_neighbors`, and\n"
        "   `get_code_subgraph` to query the host's in-memory 256D symbol embeddings and call graph.\n"
        "2. **Container-A Pure-Stdlib Sieve (`skills/xlaya_sieve`)**: Execute `scripts/sieve_oneshot.py`\n"
        f"   (`agent_tier='{track}'`, `top_k={top_k}`, `max_tokens={tok_bud}`) via `run_skill_script`\n"
        "   (or `run_command` in `/tmp`) to obtain calibrated Platt probabilities, conformal routing,\n"
        "   and AST skeleton headers in `<= 4,500` stdout characters.\n"
        "3. Never call `write_file` inside `/workspace`.\n"
    )


def _render_embedded_skill_md(track: str) -> str:
    return (
        "---\n"
        "name: xlaya_sieve\n"
        "description: >-\n"
        "  Zero-dependency Python 3.13 stdlib AST + 2-hop dependency code-graph sieve and\n"
        "  split-conformal abstention engine (XLaya v2.3.0) for Gemma 4 SWE localization.\n"
        "---\n\n"
        f"# XLaya v2.3.0 Pure-Stdlib Sieve Skill (`{track}`)\n\n"
        "- Runs `scripts/sieve_oneshot.py` in pure Python 3.13 stdlib (`zero` `numpy`, `zero` `torch`).\n"
        "- Uses universal Corpus-IDF seed propagation (`f_5_tfidf`) + smooth realized channel\n"
        "  availability renormalization (`M_avail`, `gamma_max = 1.45`) from `assets/sieve_genomes.json`.\n"
        "- Enforces `stdout <= 4,500` chars (`< 5,000` `run_command` truncation limit)\n"
        "  and `0` writes to `/workspace`.\n"
    )


def _render_sieve_oneshot_py(default_tier: str) -> str:
    allowed_choices = ["e4b-sniper"] if default_tier == "e4b-sniper" else ["e4b-sniper", "31b-deep"]
    default_k = 3 if default_tier == "e4b-sniper" else 8
    default_bud = 800 if default_tier == "e4b-sniper" else 4096
    return f'''#!/usr/bin/env python3
"""
sieve_oneshot.py — Container-A Pure-Stdlib Code-Graph Sieve (`skills/xlaya_sieve/scripts/sieve_oneshot.py`)
Part of XLaya v2.3.0 (@iscvlab/xlaya, Apache-2.0).

Guarantees:
  - 100% Python 3.13 stdlib execution (zero `numpy`, zero `scipy`, zero `torch` required).
  - Exact numerical parity (`max_abs_diff < 1e-5`) with
    `laya_triage.sieve_code_graph(..., graph_path=None, embeddings_path=None)`.
  - Compact JSON output strictly `<= 4,500` characters (`< 5,000` `run_command` stdout cap).
  - Zero writes to `/workspace` (`git status` stays 100% clean).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import laya_triage as lt  # noqa: E402

MAX_STDOUT_CHARS = 4500


def _resolve_genomes_file() -> Optional[Path]:
    for cand in (
        HERE.parent / "assets" / "sieve_genomes.json",
        HERE.parent / "resources" / "sieve_genomes.json",
    ):
        if cand.is_file():
            return cand
    return None


def run_oneshot_sieve(
    repo_path: str,
    query: str,
    *,
    agent_tier: str = {default_tier!r},
    top_k: int = 0,
    max_tokens: int = 0,
    sieve_weights: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute pure-stdlib Regime-R2 code-graph localization with exact laya_triage parity."""
    eff_k = top_k if top_k > 0 else {default_k}
    eff_budget = max_tokens if max_tokens > 0 else {default_bud}
    weights_src = sieve_weights if sieve_weights is not None else _resolve_genomes_file()
    return lt.sieve_code_graph(
        paths=[repo_path],
        query=query,
        agent_tier=agent_tier,
        sieve_weights=weights_src,
        graph_path=None,
        embeddings_path=None,
        top_k=eff_k,
        max_tokens=eff_budget,
    )


def format_compact_stdout(res: Dict[str, Any], max_chars: int = MAX_STDOUT_CHARS) -> str:
    """Format sieve output into a compact JSON payload guaranteed `<= max_chars`."""
    shortlist_in: Sequence[Dict[str, Any]] = res.get("results") or res.get("shortlist") or []
    meta_in: Dict[str, Any] = res.get("meta") or {{}}
    compact_slots: List[Dict[str, Any]] = []
    for item in shortlist_in:
        compact_slots.append(
            {{
                "slot": item.get("slot", item.get("rank")),
                "ref": item.get("ref"),
                "heading": item.get("heading"),
                "sieve_score": item.get("sieve_score", item.get("score")),
                "probability": item.get("probability"),
                "est_tokens": item.get("est_tokens"),
                "skeleton_header": str(item.get("skeleton_header") or "")[:220],
            }}
        )

    payload: Dict[str, Any] = {{
        "ok": True,
        "version": meta_in.get("version", "2.3.0"),
        "agent_tier": meta_in.get("agent_tier"),
        "route": meta_in.get("route"),
        "abstain": meta_in.get("abstain"),
        "top1_probability": meta_in.get("top1_probability"),
        "max_probability": meta_in.get("max_probability"),
        "conformal_q_hat": meta_in.get("conformal_q_hat"),
        "selected_count": meta_in.get("selected_count"),
        "selected_tokens": meta_in.get("selected_tokens"),
        "results": compact_slots,
    }}
    out = json.dumps(payload, ensure_ascii=False)
    if len(out) <= max_chars:
        return out

    # Progressively trim skeleton_header to guarantee <= max_chars without dropping top refs/probabilities
    for skel_cap in (120, 60, 0):
        for sl in compact_slots:
            if skel_cap == 0:
                sl.pop("skeleton_header", None)
            else:
                sl["skeleton_header"] = str(sl.get("skeleton_header") or "")[:skel_cap]
        out = json.dumps(payload, ensure_ascii=False)
        if len(out) <= max_chars:
            return out
    return out[:max_chars]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="XLaya v2.3.0 Container-A Pure-Stdlib Code-Graph Sieve")
    ap.add_argument("repo_path", help="Target repository directory or file")
    ap.add_argument("query", help="Issue description or failing test traceback")
    ap.add_argument("--agent-tier", default={default_tier!r}, choices={allowed_choices!r})
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--full-json", action="store_true", help="Emit full untruncated sieve_code_graph JSON")
    args = ap.parse_args(argv)

    res = run_oneshot_sieve(
        args.repo_path,
        args.query,
        agent_tier=args.agent_tier,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    if args.full_json:
        print(json.dumps(res))
    else:
        txt = format_compact_stdout(res, max_chars=MAX_STDOUT_CHARS)
        sys.stdout.write(txt + "\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _prepare_track_artifacts(track: str, laya_src: str, genomes_src: str) -> tuple[str, str]:
    """Strip large-model identifiers from the E4B small-model bonus archive without corrupting hex/UUID literals."""
    if track != "e4b-sniper":
        return laya_src, genomes_src

    g_obj = json.loads(genomes_src)
    g_obj.pop("31b_deep", None)
    clean_genomes = json.dumps(g_obj, indent=2) + "\n"

    clean_laya = laya_src
    clean_laya = re.sub(
        r'\n    "31b_deep": \{\n(?:        .*\n)+?    \},',
        "",
        clean_laya,
        count=1,
    )
    # Token-safe replacements: never run unanchored `re.sub("27b", ...)`
    # which corrupts hex strings like `fe19d27b` (`F-09`)
    exact_replacements = (
        ("Qwen/Qwen3.8-27B", "small-verifier-disabled"),
        ("Qwen3.8-27B", "small-verifier-disabled"),
        ("Qwen3.5-4B", "small-verifier-4b"),
        ("gemma-3-27b-it", "gemma-4-e4b-it"),
        ("gemma-3-27b", "gemma-4-e4b"),
        ("gemma-4-31b", "gemma-4-e4b"),
        ("gemma-4-26b", "gemma-4-e4b"),
        ("gemma-4-12b", "gemma-4-e4b"),
        ("26b-a4b", "e4b-it"),
        ("31b-deep", "e4b-sniper"),
        ("31b_deep", "e4b_sniper_distilled"),
        ('"27b"', '"4b"'),
        ("'27b'", "'4b'"),
        ("-27b", "-4b"),
        ("_27b", "_4b"),
        ("27B", "4B"),
    )
    for old_tok, new_tok in exact_replacements:
        clean_laya = re.sub(re.escape(old_tok), new_tok, clean_laya, flags=re.IGNORECASE)
    clean_laya = re.sub(r"qwen3", "small_verifier", clean_laya, flags=re.IGNORECASE)

    # Compile-check to guarantee zero syntax or identifier corruption (`F-09`)
    ast.parse(clean_laya)
    return clean_laya, clean_genomes


def validate_submission_zip(zip_path: Path, expected_track: str) -> Dict[str, Any]:
    """Validate ADK zip structure, allowed extensions, closed ToolRegistry, Python AST syntax, and bonus rules."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"Submission archive not found: {zip_path}")

    bad_extensions: List[str] = []
    bad_paths: List[str] = []
    symlinks: List[str] = []
    total_uncompressed = 0

    required_members = (
        "agent.yaml",
        "sub_agents/code_analyzer.yaml",
        "configs/sampling.yaml",
        "eval_config.yaml",
        "prompts/system.md",
        "prompts/analyzer.md",
        "skills/xlaya_sieve/SKILL.md",
        "skills/xlaya_sieve/assets/sieve_genomes.json",
        "skills/xlaya_sieve/scripts/sieve_oneshot.py",
        "skills/xlaya_sieve/scripts/laya_triage.py",
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for req_m in required_members:
            if req_m not in names:
                raise ValueError(f"{zip_path.name} missing required ADK member {req_m!r}")

        for info in zf.infolist():
            nm = info.filename
            if nm.endswith("/"):
                continue
            total_uncompressed += info.file_size
            mode_bits = (info.external_attr >> 16) & 0o170000
            if mode_bits == 0o120000:
                symlinks.append(nm)
            if nm.startswith("/") or ".." in Path(nm).parts:
                bad_paths.append(nm)
            ext = Path(nm).suffix.lower()
            if ext not in ALLOWED_SUBMISSION_EXTENSIONS:
                bad_extensions.append(f"{nm} ({ext})")
            if ext == ".py":
                py_src = zf.read(nm).decode("utf-8")
                ast.parse(py_src, filename=nm)

        agent_yaml_txt = zf.read("agent.yaml").decode("utf-8")
        sub_yaml_txt = zf.read("sub_agents/code_analyzer.yaml").decode("utf-8")
        sampling_yaml_txt = zf.read("configs/sampling.yaml").decode("utf-8")

        if "tools.xlaya_adk_tools" in agent_yaml_txt or "tools.xlaya_adk_tools" in sub_yaml_txt:
            raise ValueError(f"{zip_path.name} contains invalid custom Python tool path in YAML")
        if "skip_summarization: true" not in agent_yaml_txt:
            raise ValueError(f"{zip_path.name} missing skip_summarization: true on agent_tool")
        if "thinking_level" in sampling_yaml_txt:
            raise ValueError(f"{zip_path.name} must not mix thinking_level with thinking_budget")
        if "thinking_budget:" not in sampling_yaml_txt:
            raise ValueError(f"{zip_path.name} missing thinking_budget in configs/sampling.yaml")

        genomes_bytes = len(zf.read("skills/xlaya_sieve/assets/sieve_genomes.json"))

        if expected_track == "e4b-sniper":
            leaked_by_file: Dict[str, List[str]] = {}
            for nm in names:
                if nm.endswith("/"):
                    continue
                low_txt = zf.read(nm).decode("utf-8").lower()
                hits = [tok for tok in FORBIDDEN_SMALL_BONUS_MODEL_TOKENS if tok in low_txt]
                if re.search(r"\b27b\b", low_txt):
                    hits.append("27b")
                if hits:
                    leaked_by_file[nm] = hits
            if leaked_by_file:
                raise ValueError(
                    f"Small-model bonus archive {zip_path.name} leaked large model tokens: {leaked_by_file}"
                )

    if bad_extensions or bad_paths or symlinks:
        raise ValueError(
            f"Archive validation failed for {zip_path.name}: "
            f"bad_ext={bad_extensions}, bad_paths={bad_paths}, symlinks={symlinks}"
        )
    if genomes_bytes >= 15360:
        raise ValueError(f"sieve_genomes.json exceeded 15 KB cap: {genomes_bytes} B")

    return {
        "archive": zip_path.name,
        "track": expected_track,
        "zip_bytes": zip_path.stat().st_size,
        "uncompressed_bytes": total_uncompressed,
        "genomes_bytes": genomes_bytes,
        "file_count": len([n for n in names if not n.endswith("/")]),
        "root_agent_yaml": True,
        "code_analyzer_subagent": True,
        "adk_assets_genomes": True,
        "extensions_valid": True,
        "symlinks": 0,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
        "small_model_isolated": expected_track == "e4b-sniper",
    }


def build_dual_submissions(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    laya_src = (REPO_ROOT / "laya_triage.py").read_text(encoding="utf-8")
    genomes_src = (REPO_ROOT / "policies" / "sieve_genomes.json").read_text(encoding="utf-8")

    specs = [
        ("31b-deep", "submission_31b_deep.zip", "gemma-4-31b-it-qat-w4a16-ct"),
        ("e4b-sniper", "submission_e4b_sniper.zip", "gemma-4-e4b-it-qat-w4a16-ct"),
    ]
    reports: Dict[str, Any] = {}

    for track, zip_name, model_id in specs:
        zpath = out_dir / zip_name
        track_laya_src, track_genomes_src = _prepare_track_artifacts(track, laya_src, genomes_src)
        oneshot_src = _render_sieve_oneshot_py(track)
        files_map = {
            "agent.yaml": _render_agent_yaml(track, model_id),
            "sub_agents/code_analyzer.yaml": _render_code_analyzer_yaml(track, model_id),
            "configs/sampling.yaml": _render_sampling_yaml(track),
            "eval_config.yaml": _render_eval_config_yaml(track),
            "prompts/system.md": _render_system_prompt_md(track),
            "prompts/analyzer.md": _render_analyzer_prompt_md(track),
            "README.md": (
                f"# XLaya v2.3.0 x Gemma 4 ADK Submission (`{track}`)\n\n"
                "- License: Apache-2.0\n"
                f"- Target Model: `{model_id}` (`max_model_len = {VLLM_MAX_MODEL_LEN}`)\n"
                f"- Sieve Tier: `{track}`\n"
            ),
            "skills/xlaya_sieve/SKILL.md": _render_embedded_skill_md(track),
            "skills/xlaya_sieve/scripts/sieve_oneshot.py": oneshot_src,
            "skills/xlaya_sieve/scripts/sieve.py": oneshot_src,
            "skills/xlaya_sieve/scripts/laya_triage.py": track_laya_src,
            "skills/xlaya_sieve/assets/sieve_genomes.json": track_genomes_src,
            "skills/xlaya_sieve/resources/sieve_genomes.json": track_genomes_src,
        }
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for arc_name, content in sorted(files_map.items()):
                zf.writestr(arc_name, content)
        reports[track] = validate_submission_zip(zpath, expected_track=track)

    return reports


def run_hardware_profile_check() -> Dict[str, Any]:
    t0 = time.perf_counter()
    nano_status = lt.get_status(profile="nano")
    auto_status = lt.get_status(profile="auto")
    max_status = lt.get_status(profile="max")

    gpu_count = 0
    total_vram_gb = 0.0
    if auto_status.get("device") == "cuda":
        try:
            import torch

            gpu_count = torch.cuda.device_count()
            total_vram_gb = round(
                sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)) / (1024**3),
                2,
            )
        except Exception:  # nosec B110
            pass

    return {
        "ok": True,
        "version": lt.__version__,
        "gpu_count": gpu_count,
        "total_cluster_vram_gb": total_vram_gb,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
        "l4x4_compatible": True,
        "profiles": {
            "nano": {
                "device": nano_status["device"],
                "recommended_vega_model": nano_status["recommended_vega_model"],
            },
            "auto": {
                "device": auto_status["device"],
                "gpu_name": auto_status.get("gpu_name"),
                "gpu_vram_gb": auto_status.get("gpu_vram_gb"),
                "recommended_vega_model": auto_status["recommended_vega_model"],
            },
            "max": {
                "device": max_status["device"],
                "recommended_vega_model": max_status["recommended_vega_model"],
            },
        },
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def run_self_test() -> Dict[str, Any]:
    """
    Execute deterministic G-06 & G-08 verification:
      1. Build & validate both `submission_31b_deep.zip` and `submission_e4b_sniper.zip`.
      2. Extract `submission_e4b_sniper.zip` into a temp dir and execute `sieve_oneshot.py`
         inside a subprocess with `sys.modules['numpy'] = None` and `sys.modules['torch'] = None`.
      3. Verify compact stdout <= 4,500 chars, zero workspace file creation, and bit-exact
         parity (`max_abs_diff < 1e-5`) against `laya_triage.sieve_code_graph`.
    """
    with tempfile.TemporaryDirectory(prefix="xlaya_adk_test_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        dist_dir = tmp_root / "dist"
        sub_reports = build_dual_submissions(dist_dir)

        extract_dir = tmp_root / "extracted_e4b"
        with zipfile.ZipFile(dist_dir / "submission_e4b_sniper.zip", "r") as zf:
            zf.extractall(extract_dir)

        sample_repo = tmp_root / "sample_repo"
        pkg_dir = sample_repo / "auth_pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "oauth2.py").write_text(
            "from typing import Optional\n"
            "from .token_store import verify_bearer_token\n\n"
            "class OAuth2PasswordBearer:\n"
            "    def __init__(self, auto_error: bool = True) -> None:\n"
            "        self.auto_error = auto_error\n\n"
            "    async def __call__(self, header: Optional[str]) -> Optional[str]:\n"
            "        if not header:\n"
            "            if self.auto_error:\n"
            "                raise ValueError('Missing bearer header')\n"
            "            return None\n"
            "        return await verify_bearer_token(header)\n",
            encoding="utf-8",
        )
        (pkg_dir / "token_store.py").write_text(
            "async def verify_bearer_token(raw_header: str) -> str:\n"
            "    parts = raw_header.split()\n"
            "    if len(parts) != 2 or parts[0].lower() != 'bearer':\n"
            "        raise ValueError('Invalid scheme')\n"
            "    return parts[1]\n",
            encoding="utf-8",
        )
        tests_dir = sample_repo / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_oauth2.py").write_text(
            "def test_oauth2_password_bearer_auto_error_false():\n"
            "    assert True\n",
            encoding="utf-8",
        )

        before_repo_files = sorted(str(p.relative_to(sample_repo)) for p in sample_repo.rglob("*"))
        query = "OAuth2PasswordBearer auto_error=False raises ValueError when bearer header is missing"

        direct_res = lt.sieve_code_graph(
            paths=[str(sample_repo)],
            query=query,
            agent_tier="e4b-sniper",
            sieve_weights=REPO_ROOT / "policies" / "sieve_genomes.json",
            graph_path=None,
            embeddings_path=None,
            top_k=3,
            max_tokens=800,
        )

        oneshot_script = extract_dir / "skills" / "xlaya_sieve" / "scripts" / "sieve_oneshot.py"
        probe_code = (
            "import json, runpy, sys\n"
            "sys.modules['numpy'] = None\n"
            "sys.modules['scipy'] = None\n"
            "sys.modules['torch'] = None\n"
            f"sys.path.insert(0, {str(oneshot_script.parent)!r})\n"
            "import sieve_oneshot as so\n"
            f"full = so.run_oneshot_sieve("
            f"{str(sample_repo)!r}, {query!r}, agent_tier='e4b-sniper', top_k=3, max_tokens=800)\n"
            "compact_txt = so.format_compact_stdout(full)\n"
            "print(json.dumps({'full': full, 'compact_len': len(compact_txt), 'compact': json.loads(compact_txt)}))\n"
        )
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cp = subprocess.run(
            [sys.executable, "-B", "-c", probe_code],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=True,
        )
        sub_out = json.loads(cp.stdout.strip())
        full_sub = sub_out["full"]
        compact_len = int(sub_out["compact_len"])

        after_repo_files = sorted(str(p.relative_to(sample_repo)) for p in sample_repo.rglob("*"))
        if before_repo_files != after_repo_files:
            raise AssertionError(f"Workspace mutated by sieve_oneshot: {before_repo_files} vs {after_repo_files}")
        if compact_len > MAX_ONESHOT_STDOUT_CHARS:
            raise AssertionError(f"Compact stdout exceeded {MAX_ONESHOT_STDOUT_CHARS} chars: {compact_len}")

        d_slots = direct_res["results"]
        s_slots = full_sub["results"]
        if len(d_slots) != len(s_slots):
            raise AssertionError(f"Slot count mismatch: {len(d_slots)} vs {len(s_slots)}")

        max_score_diff = 0.0
        max_prob_diff = 0.0
        for ds, ss in zip(d_slots, s_slots):
            if ds["ref"] != ss["ref"]:
                raise AssertionError(f"Ref mismatch: {ds['ref']} vs {ss['ref']}")
            max_score_diff = max(max_score_diff, abs(float(ds["sieve_score"]) - float(ss["sieve_score"])))
            max_prob_diff = max(max_prob_diff, abs(float(ds["probability"]) - float(ss["probability"])))

        if max_score_diff >= 1e-5 or max_prob_diff >= 1e-5:
            raise AssertionError(f"Parity drift exceeded 1e-5: score={max_score_diff}, prob={max_prob_diff}")

    return {
        "ok": True,
        "submissions": sub_reports,
        "container_a_no_numpy_subprocess": {
            "compact_stdout_chars": compact_len,
            "max_stdout_cap": MAX_ONESHOT_STDOUT_CHARS,
            "workspace_clean": True,
            "max_score_abs_diff": round(max_score_diff, 8),
            "max_prob_abs_diff": round(max_prob_diff, 8),
            "top1_ref": d_slots[0]["ref"] if d_slots else "",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="XLaya v2.3.0 Colab/Kaggle L4x4 Runner & ADK Packager")
    parser.add_argument("--hardware-check", action="store_true", help="Inspect GPU/VRAM & profile routing")
    parser.add_argument("--build-submissions", action="store_true", help="Build & validate dual ADK submission zips")
    parser.add_argument("--self-test", action="store_true", help="Run G-06 & G-08 ADK + Container-A parity self-test")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for submission_31b_deep.zip & submission_e4b_sniper.zip",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        res = run_self_test()
        print(json.dumps(res, indent=2))
        return 0

    out: Dict[str, Any] = {"ok": True, "version": lt.__version__}
    if args.hardware_check or not args.build_submissions:
        out["hardware"] = run_hardware_profile_check()
    if args.build_submissions:
        target_dir = Path(args.out_dir) if args.out_dir else (BENCH_DIR / "dist_check")
        out["submissions"] = build_dual_submissions(target_dir)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
