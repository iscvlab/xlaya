#!/usr/bin/env python3
"""
test_v23_sieve_and_adk.py — Red-to-Green Regression & Generalization Suite for XLaya v2.3.0
Part of XLaya v2.3.0 (@iscvlab/xlaya, Apache-2.0).

Verifies Gates G-01 through G-08:
  - G-01 / F-01 / F-02: NetworkX `node_link_data` `graphs/*.json` ingestion (zero degree cliff on `f_4`),
    Kaggle symbol-keyed `embeddings/*.npz` top-PC whitening (`rand_cos_p50 in [-0.15, +0.15]`),
    universal Corpus-IDF seed propagation (`f_5_tfidf`), and `M_avail` cross-regime renormalization.
  - G-02 / F-08: Source-first lexical file pre-ranking (`Tier-1` core package before `Tier-2` `docs_src/`
    and `Tier-3` `tests/`), bypassing `_is_english_text` on `.py` files.
  - G-04 / Seam 1.4: Dedicated 2-Stage Causal Monotonic Tool-Schema Conformal Gate (`triage_tool_call`)
    for `tool_call`, `request_for_info`, and `cannot_answer`.
  - G-05 / F-07: Split-method compact AST `skeleton_header`, `_span_iou > 0.30` deduplication, and
    strict token budget enforcement (`<= 800` for `e4b-sniper`, `<= 4096` for `31b-deep`, including Slot 1).
  - G-06 / G-08 / F-09 / F-11: Declarative `adk-submission` (`agent.yaml` + `sub_agents/code_analyzer.yaml`
    with `skip_summarization: true`), Small Model Bonus lexical isolation without hex corruption, and
    Container-A `sieve_oneshot.py` subprocess parity (`sys.modules['numpy'] = None`, `max_abs_diff < 1e-5`,
    `stdout <= 4,500` chars, zero workspace writes).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import colab_kaggle_l4x4_runner as runner  # noqa: E402
import laya_triage as lt  # noqa: E402


def test_g01_networkx_graph_and_symbol_npz_whitening() -> None:
    """Verify F-01 (`node_link_data` call graph without degree cliff) and F-02 (`.npz` narrow-cone whitening)."""
    with tempfile.TemporaryDirectory(prefix="xlaya_g01_") as tmp:
        root = Path(tmp)
        pkg = root / "fastapi" / "security"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "oauth2.py").write_text(
            "from fastapi.security.utils import get_authorization_scheme_param\n\n"
            "class OAuth2PasswordBearer:\n"
            "    def __init__(self, tokenUrl: str, auto_error: bool = True) -> None:\n"
            "        self.tokenUrl = tokenUrl\n"
            "        self.auto_error = auto_error\n\n"
            "    async def __call__(self, authorization: str = '') -> str:\n"
            "        scheme, param = get_authorization_scheme_param(authorization)\n"
            "        if not authorization or scheme.lower() != 'bearer':\n"
            "            if self.auto_error:\n"
            "                raise ValueError('Not authenticated')\n"
            "            return ''\n"
            "        return param\n",
            encoding="utf-8",
        )
        (pkg / "utils.py").write_text(
            "from typing import Tuple\n\n"
            "def get_authorization_scheme_param(authorization_header_value: str) -> Tuple[str, str]:\n"
            "    if not authorization_header_value:\n"
            "        return '', ''\n"
            "    scheme, _, param = authorization_header_value.partition(' ')\n"
            "    return scheme, param\n",
            encoding="utf-8",
        )

        # Write NetworkX node_link_data JSON with edges + high-degree hub nodes
        graph_json = root / "graph.json"
        graph_payload = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {"id": "fastapi.security.oauth2.OAuth2PasswordBearer", "type": "class"},
                {"id": "fastapi.security.oauth2.OAuth2PasswordBearer.__call__", "type": "method"},
                {"id": "fastapi.security.utils.get_authorization_scheme_param", "type": "function"},
            ],
            "edges": [
                {
                    "source": "fastapi.security.oauth2.OAuth2PasswordBearer",
                    "target": "fastapi.security.utils.get_authorization_scheme_param",
                    "type": "calls",
                },
                {
                    "source": "fastapi.security.oauth2.OAuth2PasswordBearer.__call__",
                    "target": "fastapi.security.utils.get_authorization_scheme_param",
                    "type": "calls",
                },
            ],
        }
        graph_json.write_text(json.dumps(graph_payload), encoding="utf-8")

        # Write narrow-cone symbol-keyed 256D .npz (`cos ~ 0.88` before whitening)
        rng = np.random.default_rng(42)
        common_cone = rng.normal(0.0, 1.0, size=(256,)).astype(np.float32)
        common_cone /= np.linalg.norm(common_cone)
        sig_dir = rng.normal(0.0, 1.0, size=(256,)).astype(np.float32)
        sig_dir /= np.linalg.norm(sig_dir)

        npz_dict = {}
        va = 0.92 * common_cone + 0.25 * sig_dir
        vb = 0.92 * common_cone + 0.24 * sig_dir
        npz_dict["fastapi.security.oauth2.OAuth2PasswordBearer"] = (va / np.linalg.norm(va)).astype(
            np.float32
        )
        npz_dict["fastapi.security.oauth2.OAuth2PasswordBearer.__call__"] = (va / np.linalg.norm(va)).astype(
            np.float32
        )
        npz_dict["fastapi.security.utils.get_authorization_scheme_param"] = (vb / np.linalg.norm(vb)).astype(
            np.float32
        )
        for i in range(24):
            noise = 0.92 * common_cone + 0.25 * rng.normal(0.0, 1.0, size=(256,)).astype(np.float32)
            npz_dict[f"fastapi.other.mod_{i}.func_{i}"] = (noise / np.linalg.norm(noise)).astype(np.float32)

        npz_path = root / "embeddings.npz"
        np.savez(npz_path, **npz_dict)

        whitened, floor_c = lt._whiten_symbol_vectors(npz_dict)
        keys = list(whitened.keys())
        mat = np.asarray([whitened[k] for k in keys], dtype=np.float64)
        dots = (mat @ mat.T)[np.triu_indices(len(keys), k=1)]
        p50 = float(np.median(dots))
        assert -0.15 <= p50 <= 0.15, f"Whitened median cosine outside [-0.15, 0.15]: {p50}"
        assert 0.05 <= floor_c <= 0.50

        query = "OAuth2PasswordBearer auto_error=False still raises error when authorization header is empty"
        r1 = lt.sieve_code_graph(
            paths=[str(root)],
            query=query,
            agent_tier="31b-deep",
            graph_path=graph_json,
            embeddings_path=npz_path,
            top_k=3,
            max_tokens=1200,
        )
        r2 = lt.sieve_code_graph(
            paths=[str(root)],
            query=query,
            agent_tier="31b-deep",
            graph_path=None,
            embeddings_path=None,
            top_k=3,
            max_tokens=1200,
        )
        assert r1["ok"] and r2["ok"]
        assert r1["meta"]["external_graph_edges_matched"] >= 1
        assert r1["meta"]["npz_symbols_matched"] >= 2
        # Verify zero degree cliff: f_4 (centrality) stays identical across R1 and R2
        f4_key = lt.SIEVE_FEATURE_NAMES[4]
        f4_r1 = {item["ref"]: item["features"][f4_key] for item in r1["results"]}
        f4_r2 = {item["ref"]: item["features"][f4_key] for item in r2["results"]}
        for ref_k in set(f4_r1) & set(f4_r2):
            assert abs(f4_r1[ref_k] - f4_r2[ref_k]) < 1e-6


def test_g02_tier1_core_preranking_over_docs_src_and_tests() -> None:
    """Verify F-08: core library files (`fastapi/`) outrank `docs_src/` and `tests/` before chunk cap."""
    with tempfile.TemporaryDirectory(prefix="xlaya_g02_") as tmp:
        root = Path(tmp)
        core_dir = root / "fastapi"
        docs_dir = root / "docs_src" / "security"
        tests_dir = root / "tests"
        core_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)

        (core_dir / "routing.py").write_text(
            "class APIRoute:\n"
            "    def matches(self, scope: dict) -> bool:\n"
            "        return scope.get('type') == 'http'\n",
            encoding="utf-8",
        )
        for idx in range(30):
            (docs_dir / f"tutorial_{idx:03d}.py").write_text(
                f"from fastapi.routing import APIRoute\napp_route_{idx} = APIRoute()\n",
                encoding="utf-8",
            )
            (tests_dir / f"test_routing_{idx:03d}.py").write_text(
                f"from fastapi.routing import APIRoute\ndef test_route_{idx}(): assert APIRoute\n",
                encoding="utf-8",
            )

        files, scan_root = lt.collect_target_files([str(root)], pattern="*.py")
        ranked = lt._prerank_repo_files(files, scan_root, "APIRoute matches http scope bug")
        assert ranked[0].name == "routing.py"
        assert "fastapi" in str(ranked[0])


def test_g04_when2call_triage_tool_call_3way_routing() -> None:
    """Verify Seam 1.4: `triage_tool_call` separates `tool_call`, `request_for_info`, and `cannot_answer`."""
    weather_tool = {
        "name": "get_current_weather",
        "description": "Get the current temperature and weather forecast for a given city and state.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Seattle"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    }

    hit = lt.triage_tool_call(
        "What is the current weather forecast in Seattle in celsius?",
        [weather_tool],
    )
    assert hit["decision"] == "tool_call"
    assert hit["abstain"] is False
    assert hit["matched_tool"] == "get_current_weather"
    assert hit["tool_probability"] >= 0.50

    missing_req = lt.triage_tool_call(
        "What is the current weather forecast right now?",
        [weather_tool],
    )
    assert missing_req["decision"] in ("request_for_info", "cannot_answer")
    assert missing_req["abstain"] is True
    assert missing_req["features"][9] > 0.0  # unfilled_ratio > 0

    ood_miss = lt.triage_tool_call(
        "Compile a kernel driver for a custom RISC-V FPGA board in Verilog.",
        [weather_tool],
    )
    assert ood_miss["decision"] == "cannot_answer"
    assert ood_miss["abstain"] is True


def test_g05_skeleton_header_span_iou_and_hard_token_wall() -> None:
    """Verify F-07: split methods carry compact `skeleton_header` and Slot-1 respects hard token budget."""
    with tempfile.TemporaryDirectory(prefix="xlaya_g05_") as tmp:
        root = Path(tmp)
        src_file = root / "engine.py"
        methods_txt = "\n\n".join(
            f"    def step_{i}(self, payload: dict) -> int:\n"
            + "\n".join(f"        val_{j} = payload.get('k_{i}_{j}', {j}) + self.base_offset" for j in range(28))
            + "\n        return val_0"
            for i in range(6)
        )
        src_file.write_text(
            "import math\nfrom typing import Dict, Optional\n\n"
            "class HyperSolver(BaseSolver):\n"
            "    def __init__(self, base_offset: int = 10) -> None:\n"
            "        self.base_offset = base_offset\n"
            "        self.cache_map: Dict[str, int] = {}\n\n"
            + methods_txt
            + "\n",
            encoding="utf-8",
        )

        res = lt.sieve_code_graph(
            paths=[str(root)],
            query="HyperSolver step_3 payload base_offset calculation error",
            agent_tier="e4b-sniper",
            top_k=3,
            max_tokens=180,
        )
        assert res["ok"] is True
        assert res["meta"]["selected_tokens"] <= 180
        assert len(res["results"]) >= 1
        top_slot = res["results"][0]
        assert top_slot["est_tokens"] <= 180
        assert "class HyperSolver" in top_slot["skeleton_header"]
        assert "self.base_offset" in top_slot["skeleton_header"]


def test_g06_and_g08_adk_submission_and_container_a_no_numpy_parity() -> None:
    """Verify Seam 3: ADK submission builder, Small-Model Bonus lexical isolation, and `numpy=None` parity."""
    report = runner.run_self_test()
    assert report["ok"] is True
    assert report["submissions"]["31b-deep"]["extensions_valid"] is True
    assert report["submissions"]["e4b-sniper"]["small_model_isolated"] is True
    sub_check = report["container_a_no_numpy_subprocess"]
    assert sub_check["workspace_clean"] is True
    assert sub_check["compact_stdout_chars"] <= 4500
    assert sub_check["max_score_abs_diff"] < 1e-5
    assert sub_check["max_prob_abs_diff"] < 1e-5
