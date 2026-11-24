#!/usr/bin/env python3
"""
test_mcp_annotations.py — MCP 2025-03-26 Protocol Negotiation, ToolAnnotations & Stdio Dispatch Tests.

Verifies:
  - JSON-RPC `initialize` protocol version negotiation (`2025-03-26` default/newer/unknown,
    explicit `2024-11-05` fallback) both in-process and over live `stdio` subprocess.
  - `tools/list` schema & explicit boolean `ToolAnnotations` (`readOnlyHint`, `destructiveHint`,
    `idempotentHint`, `openWorldHint`) on all 6 tools with bidirectional `A_pos` & `A_neg` counter-anchors.
  - Live `xlaya_status` JSON-RPC `tools/call` dispatch + negative unknown-tool counter-anchor.
  - `--self-check` CLI contract and `main()` stream/timer resource restoration.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xlaya_mcp_server as xmcp  # noqa: E402

EXPECTED_TOOL_ORDER = [
    "xlaya_status",
    "xlaya_triage",
    "xlaya_predict",
    "xlaya_fit",
    "xlaya_preload",
    "xlaya_unload",
]

REQUIRED_ANNOTATION_KEYS = (
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)

EXPECTED_ANNOTATIONS: Dict[str, Dict[str, bool]] = {
    "xlaya_status": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "xlaya_triage": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
    "xlaya_predict": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
    "xlaya_fit": {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
    "xlaya_preload": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
    "xlaya_unload": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}


def _run_stdio_session(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run `xlaya_mcp_server.py` over live stdio and parse every stdout JSON-RPC frame."""
    stdin_payload = "\n".join(json.dumps(frame) for frame in frames) + "\n"
    cp = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "xlaya_mcp_server.py")],
        input=stdin_payload,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


@pytest.mark.parametrize(
    "params,expected_version",
    [
        ({}, "2025-03-26"),
        ({"protocolVersion": "2025-03-26"}, "2025-03-26"),
        ({"protocolVersion": "2025-06-18"}, "2025-03-26"),
        ({"protocolVersion": "2099-01-01"}, "2025-03-26"),
        ({"protocolVersion": "unknown-version"}, "2025-03-26"),
        ({"protocolVersion": ""}, "2025-03-26"),
        ({"protocolVersion": 20241105}, "2025-03-26"),
        ({"protocolVersion": "2024-11-05"}, "2024-11-05"),
        ({"protocolVersion": " 2024-11-05 "}, "2024-11-05"),
    ],
)
def test_initialize_protocol_negotiation_in_process(params: Dict[str, Any], expected_version: str) -> None:
    """Verify MCP protocol negotiation (`2025-03-26` default/newer/unknown; `2024-11-05` fallback)."""
    resp = xmcp.handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params}
    )
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == expected_version
    assert result["capabilities"] == {"tools": {}}
    assert result["serverInfo"] == {"name": "xlaya-mcp", "version": xmcp.__version__}


def test_initialize_non_dict_params_defaults_safely() -> None:
    """Non-object `params` on `initialize` falls back cleanly to latest protocol version."""
    resp = xmcp.handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 99, "method": "initialize", "params": None}
    )
    assert resp is not None
    assert resp["result"]["protocolVersion"] == "2025-03-26"


def test_tools_list_annotations_strict_booleans_and_counter_anchors() -> None:
    """Verify all 6 tools expose 4 explicit bool hints with bidirectional `A_pos` & `A_neg` checks."""
    resp = xmcp.handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert resp is not None
    tools = resp["result"]["tools"]
    assert len(tools) == 6
    assert [t["name"] for t in tools] == EXPECTED_TOOL_ORDER

    by_name: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        name = tool["name"]
        by_name[name] = tool
        annotations = tool.get("annotations")
        assert isinstance(annotations, dict), f"Missing annotations dict on {name}"
        assert set(annotations.keys()) == set(REQUIRED_ANNOTATION_KEYS), (
            f"Unexpected annotation keys on {name}: {set(annotations.keys())}"
        )
        for key in REQUIRED_ANNOTATION_KEYS:
            val = annotations[key]
            assert type(val) is bool, f"Non-bool annotation {name}.{key}={val!r} ({type(val).__name__})"
        assert annotations == EXPECTED_ANNOTATIONS[name]

    # Bidirectional A_pos & A_neg counter-anchors
    read_only_tools = {n for n, t in by_name.items() if t["annotations"]["readOnlyHint"] is True}
    mutating_tools = {n for n, t in by_name.items() if t["annotations"]["readOnlyHint"] is False}
    assert read_only_tools == {"xlaya_status", "xlaya_triage", "xlaya_predict"}
    assert mutating_tools == {"xlaya_fit", "xlaya_preload", "xlaya_unload"}
    assert read_only_tools.isdisjoint(mutating_tools)

    destructive_tools = {n for n, t in by_name.items() if t["annotations"]["destructiveHint"] is True}
    non_destructive_tools = {n for n, t in by_name.items() if t["annotations"]["destructiveHint"] is False}
    assert destructive_tools == {"xlaya_fit"}
    assert non_destructive_tools == {
        "xlaya_status",
        "xlaya_triage",
        "xlaya_predict",
        "xlaya_preload",
        "xlaya_unload",
    }

    open_world_tools = {n for n, t in by_name.items() if t["annotations"]["openWorldHint"] is True}
    closed_world_tools = {n for n, t in by_name.items() if t["annotations"]["openWorldHint"] is False}
    assert open_world_tools == {"xlaya_triage", "xlaya_predict", "xlaya_fit", "xlaya_preload"}
    assert closed_world_tools == {"xlaya_status", "xlaya_unload"}

    assert all(t["annotations"]["idempotentHint"] is True for t in by_name.values())


def test_live_stdio_initialize_tools_list_and_status_dispatch() -> None:
    """Verify live stdio JSON-RPC framing for `2025-03-26`, `2024-11-05`, `tools/list`, and `xlaya_status`."""
    frames = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "xlaya_status", "arguments": {"profile": "nano"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "xlaya_nonexistent_tool", "arguments": {}},
        },
    ]
    responses = _run_stdio_session(frames)
    assert len(responses) == 5
    by_id = {r["id"]: r for r in responses}

    assert by_id[1]["result"]["protocolVersion"] == "2025-03-26"
    assert by_id[2]["result"]["protocolVersion"] == "2024-11-05"

    tools = by_id[3]["result"]["tools"]
    assert len(tools) == 6
    for tool in tools:
        ann = tool["annotations"]
        assert all(type(ann[k]) is bool for k in REQUIRED_ANNOTATION_KEYS)
        assert ann == EXPECTED_ANNOTATIONS[tool["name"]]

    status_result = by_id[4]["result"]
    assert status_result["isError"] is False
    assert len(status_result["content"]) == 1
    assert status_result["content"][0]["type"] == "text"
    status_payload = json.loads(status_result["content"][0]["text"])
    assert status_payload["ok"] is True
    assert status_payload["mcp_version"] == xmcp.__version__
    assert isinstance(status_payload["idle_ttl_sec"], int) and status_payload["idle_ttl_sec"] >= 0

    err_result = by_id[5]["result"]
    assert err_result["isError"] is True
    err_payload = json.loads(err_result["content"][0]["text"])
    assert err_payload["ok"] is False
    assert "Unknown MCP tool" in err_payload["error"]


def test_self_check_cli_and_in_process_main_resource_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `--self-check` CLI output and that in-process `main(['--self-check'])` restores `sys.stdout`."""
    cp = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "xlaya_mcp_server.py"), "--self-check"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["ok"] is True
    assert summary["server"] == "xlaya-mcp"
    assert summary["version"] == xmcp.__version__
    assert summary["protocolVersion"] == "2025-03-26"
    assert summary["supportedProtocolVersions"] == ["2025-03-26", "2024-11-05"]
    assert summary["tools"] == EXPECTED_TOOL_ORDER
    assert summary["annotations_verified"] is True

    fake_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_out)
    rc = xmcp.main(["--self-check"])
    assert rc == 0
    assert sys.stdout is fake_out
    assert xmcp._IDLE_TIMER is None
    inproc_summary = json.loads(fake_out.getvalue().strip())
    assert inproc_summary["ok"] is True
    assert inproc_summary["protocolVersion"] == "2025-03-26"
