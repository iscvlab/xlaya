# XLaya (`/xlaya`) - 3-Stage System 1 → System 1.5 Semantic Triage, Joint NLI Contradiction Triad & Conformal Decision Sidecar

[![Release](https://img.shields.io/badge/Release-v2.0.0-121417?style=flat-square)](https://github.com/iscvlab/xlaya)
[![License: MIT](https://img.shields.io/badge/License-MIT-059669?style=flat-square)](./LICENSE)
[![Warm MCP](https://img.shields.io/badge/Warm%20MCP-~30%20ms-D97706?style=flat-square)](#2-empirical-4-axis-benchmarks-rtx-5060-ti-16-gb-vram)
[![Batch Speed](https://img.shields.io/badge/Batch%20Speed-859%20q%2Fs%20(8.9x)-2563EB?style=flat-square)](#axis-1-cold-vs-resident-warm-mcp--official-predict_batch-throughput)
[![Engines](https://img.shields.io/badge/Engines-Laya%20v0.4.31%20%2B%20VegaML%20v0.16.1-7C3AED?style=flat-square)](https://github.com/NandhaKishorM/laya)
[![Watcher Gate](https://img.shields.io/badge/Watcher%20Gate-0.00%25%20False--Confident%20(XERJ%20%231108)-059669?style=flat-square)](#axis-3-conformal-calibration-joint-nli-contradiction-triad--xerj-1108-watcher-gate)
[![Agents](https://img.shields.io/badge/Agents-Any%20AI%20Agent-8A2BE2?style=flat-square)](#4-universal-installation--repository-layout)
[![Lab](https://img.shields.io/badge/Lab-ISCV%20%7C%20Isocurve%20Research%20Lab%20(%40iscvlab)-121417?style=flat-square)](https://github.com/iscvlab)

**XLaya** (`iscvlab/xlaya`, slash command `/xlaya`, legacy alias `/laya`) is a non-autoregressive **3-Stage System 1 → System 1.5 Semantic Decision, Joint NLI Contradiction Triad & Conformal Triage Sidecar** engineered by **[ISCV | Isocurve Research Lab (`@iscvlab`)](https://github.com/iscvlab)**. Powered by **[`laya 0.4.31+`](https://github.com/NandhaKishorM/laya)** (`ModernBERT-large` & `mmBERT`, `output_tokens: 0`) and **`vega-U` (`vegaml 0.16.1+`)**, it runs as a **warm resident JSON-RPC 2.0 `stdio` MCP server (`xlaya_mcp_server.py`, `~30 ms` warm inference)**, a **standalone Python CLI (`laya_triage.py`)**, and a **portable AI Agent Skill (`skills/xlaya/SKILL.md`)** across **any AI coding agent — cloud, IDE, CLI, or local LLM** (`Claude`, `Grok`, `Gemini / Antigravity`, `Cursor`, `Codex`, `Ollama`, `OpenHands`, `Cline`, etc.).

---

## ⚡ TL;DR - Performance & Accuracy at a Glance (`v2.0.0`)

- **Zero-Token System 1 → System 1.5 Cascade (`output_tokens: 0`)**: Prunes **`99.35%` (`152.7×`)** of low-signal repository tokens before expensive System 2 LLM reading (`173,666` → `1,137` tokens in `170.3 ms`), cutting cold 32-chunk CLI sweeps by **`7.92×` (`74.51 s` → `9.41 s`)** and reaching **`29.99 ms` single-state / `662.5 ms` 32-chunk (`96`-decision) warm latency** via the resident MCP server.
- **Batched Cross-Lingual Determinism (`859.1 q/s`, `0 / 432` Flips)**: Length-sorted `predict_batch` accelerates mixed `50% EN / 50% HI` routing by **`8.9×` (`837.9 q/s`)** and `mmBERT` multilingual batches to **`859.1 q/s` (`1.16 ms/question`, `5.9×` at `B = 8`; up to `6.5×` at `B = 16`)** with **`0` discrete label flips** across `432` paired comparisons and **`5 / 5` (`100%`)** routing across `EN`, `DE`, `HI`, `JA`, and `TH`.
- **Calibrated Contradiction & Conformal Verification**: Combines a **3-Question Joint NLI Contradiction Triad** (**`8 / 8 = 100%`** counter-anchor accuracy, **`16.6×`** OOD conflict separation, `0` false positives) with **`vega-U` conformal prediction sets** (**`90.0%`** held-out team accuracy after a `7.58 s` 40-row fit) and **XERJ `#1108` Watcher Boundary gating** (**`0.00%` wrong-and-confident fires** vs `24.49%` baseline FAIL).
- **Which `/xlaya` Execution Tier Should You Choose?**
  - **Stage 0 — Pure-Python Structural & Lexical Sieve (`0.15 ms/chunk`, `6,841 chunks/s`)**: Best for **zero-dependency CI environments**, `--lexical-prefilter` candidate pruning across `1,000+` chunks, and exact 4-backtick fence-safe `#L<start>-L<end>` breadcrumb chunking across `.md`, `.html`, `.txt`, `.json`, `.jsonl`, and `.csv`.
  - **Stage 1 — Resident MCP `Laya` (`ModernBERT-large` & `mmBERT`, `~30 ms` warm / `859 q/s`)**: **Default daily driver** for `shortlist`, `contradiction` (Joint NLI Triad), `rubric` (`0..4` completeness), `classify` (including `predict_tournament` on `>20` labels), and bilingual Thai/CJK/multilingual triage.
  - **Stage 2 — `vega-U` (`0.8B` / `4B` / `gemma-1b`) Conformal Verifier (up to `73,728` tokens & `90.0%` 40-row fitted accuracy)**: Opt-in `--vega` / `vega: true` escalation for **borderline top-K finalists**, **whole-file long-document audits (`--whole-file`)**, and calibrated `prediction_set` + `abstain` + `route` (`"auto"` vs `"review"`) gating.

> **One-Prompt Universal Agent Install** — Copy and paste into **any AI coding agent — cloud, IDE, CLI, or local LLM** (`Claude`, `Grok`, `Gemini`, `Cursor`, `Codex`, `Ollama`, `OpenHands`, `Cline`, etc.):
> ```text
> Clone https://github.com/iscvlab/xlaya and install skills/xlaya into my active agent skills directory (Windows: ./scripts/install.ps1 -All; macOS/Linux: ./scripts/install.sh --all; or copy skills/xlaya into ~/.gemini/config/skills/xlaya, ~/.claude/skills/xlaya, ~/.codex/skills/xlaya, or ~/.agents/skills/xlaya), then run python benchmarks/run_xlaya_benchmark.py --mode quick to verify /xlaya is ready.
> ```

### Horizontal Bar Comparison (`RTX 5060 Ti 16 GB VRAM` • `Laya v0.4.31 × VegaML v0.16.1`)

| Evaluation Dimension | Configuration / Engine | Visual Comparison Bar | Measured Result & Delta vs. Baseline |
| :--- | :--- | :--- | :--- |
| **1. Cold CLI Boot Speed ↑**<br>*(32-Chunk Contradiction Sweep)* | **`xlaya v2.0.0` (`predict_batch` + Local Snap)**<br>`xlaya v1.0` Sequential CLI Baseline | `████████████████████` **`7.92× Faster`**<br>`███░░░░░░░░░░░░░░░░░` `1.0× Baseline` | **`9.41 s` (`8,459.7 ms` engine)** ✨<br>`74.51 s` (`-87.4%` cold wall time) |
| **2. Mixed-Language Batch Speed ↑**<br>*(50% EN / 50% HI, `32` States × `3` Qs)* | **`Router.predict_batch` (`sort_by_length`)**<br>Sequential `predict()` Loop | `████████████████████` **`8.9× Faster`**<br>`██░░░░░░░░░░░░░░░░░░` `1.0× Loop` | **`114.6 ms` (`837.9 q/s`, `0` flips)** ✨<br>`1,016.2 ms` (`279.3 states/s` batched) |
| **3. Peak Encoder Throughput ↑**<br>*(`8` States × `4` Questions/State)* | **`mmBERT` Multilingual Batched**<br>`ModernBERT-large` English Batched<br>Sequential `mmBERT` Loop | `████████████████████` **`859.1 q/s (5.9×)`**<br>`████████░░░░░░░░░░░░` `329.8 q/s (2.7×)`<br>`███░░░░░░░░░░░░░░░░░` `146.5 q/s (1.0×)` | **`1.16 ms/q` (`37.2 ms` vs `218.5 ms`)** ✨<br>`3.03 ms/q` (`97.0 ms` vs `263.8 ms`)<br>Unbatched (`6.5×` speedup at `B = 16`) |
| **4. Few-Shot Conformal Accuracy ↑**<br>*(`routing_labels.csv` Held-Out `N=20`)* | **`vega-U 0.8B` Fitted (`40` rows, `7.58 s`)**<br>`laya` English Zero-Shot<br>`laya-typed-decisions` Zero-Shot<br>Cold `vega-U 0.8B` Zero-Shot | `██████████████████░░` **`90.0% (18/20)`**<br>`████████████████░░░░` `80.0% (16/20)`<br>`███████████████░░░░░` `75.0% (15/20)`<br>`████████████░░░░░░░░` `60.0% (12/20)` | **`+30.0 pts` over cold (`ECE 0.214`)** ✨<br>`31.73 ms` median zero-shot<br>`25.75 ms` median (`ECE 0.188`)<br>`64.36 ms` (`auto: 8`, `review: 12`) |
| **5. Contradiction & Watcher Gate ↑**<br>*(NLI Triad + `xerj-org/xerj#1108`)* | **`/xlaya` 3-Question NLI Triad**<br>**XERJ `#1108` Gate (`100% - FalseConf`)**<br>XERJ `#1108` Baseline (`24/98` FAIL) | `████████████████████` **`100.0% (8/8)`**<br>`████████████████████` **`100.0% (0/30 False)`**<br>`███████████████░░░░░` `75.51% (24.49% FAIL)` | **`16.6×` OOD separation (`0` FP)** ✨<br>**`0.00%` False-Confident (`80%` ens.)** ✨<br>`24 / 98` (`24.49%`) wrong at `p ≥ 0.80` |
| **6. Pass-B Token Compression ↑**<br>*(`27` Specs / `1,165` Chunks)* | **`/xlaya` Top-5 Section Shortlist**<br>Full-Corpus Unfiltered Read | `████████████████████` **`99.35% Saved (152.7×)`**<br>`░░░░░░░░░░░░░░░░░░░░` `0.0% Saved (1.0×)` | **`1,137` tok in `170.3 ms` (`6.8k/s`)** ✨<br>`173,666` tok across `27` specifications |

---

## 1. Architecture & 3-Stage System 1 → System 1.5 Ecosystem (`v2.0.0`)

### A. 3-Stage Cascade (`Stage 0 Sieve → Stage 1 Laya Encoder → Stage 2 Vega-U Conformal Verifier`)

```mermaid
%%{init: {'theme': 'neutral'}}%%
flowchart LR
    subgraph Canvas [" "]
        In["Input Corpus (.md .html .txt .json .jsonl .csv)"] --> S0["Stage 0: Fence-Safe Sieve + BM25 and Thai 3-Gram"]

        subgraph Stage1 ["Stage 1: Laya v0.4.31 System 1 (output_tokens: 0)"]
            Guard{"Head-Language Guard"}
            EN["ModernBERT-large (EN 512 tok)"]
            ML["mmBERT Multilingual (up to 8192 tok)"]
            Triad["Joint NLI Triad + Tournament (>20 Labels)"]
        end

        S0 --> Guard
        Guard -->|"EN State and EN Head"| EN
        Guard -->|"Non-EN State or Head"| ML
        EN --> Triad
        ML --> Triad

        Triad -->|"Borderline or Top-K (vega: true)"| S2["Stage 2: vega-U System 1.5 (up to 73728 tok)"]
        Triad -->|"Calibrated Spans (auto)"| Out["Exact file#Lstart-Lend + Conformal Route"]
        S2 -->|"prediction_set + abstain + route"| Out
    end
    classDef default fill:#ffffff,stroke:#475569,color:#0f172a,stroke-width:1px
    style Canvas fill:#ffffff,stroke:#cbd5e1,stroke-width:1.5px,color:#0f172a
    style Stage1 fill:#f8fafc,stroke:#94a3b8,color:#0f172a
```

### B. Dual Execution Path (`Resident MCP Server` vs. `Standalone CLI` + Chained `/x*` Skills)

```mermaid
%%{init: {'theme': 'neutral'}}%%
flowchart LR
    subgraph Canvas [" "]
        subgraph Hosts ["1. Entry Points and Chained /x* Skills"]
            Chain["Chained Skills (/xsync /xbrief /xdeep-research /xpublish)"]
            Direct["Standalone Agent (/xlaya) or CI Pipeline"]
        end

        subgraph Paths ["2. Dual Execution Engine"]
            MCP["Tier 1: xlaya_mcp_server.py (~30 ms Warm + Idle Unload)"]
            CLI["Tier 2: laya_triage.py (7.92x Faster Batch CLI)"]
            Fit["Automated Conformal Fit Daemon (coming soon)"]
        end

        subgraph Actions ["3. System 2 Surgical Action"]
            Skip["Skip Clean Chunks (Zero Token Bloat)"]
            Edit["Surgical read_file and edit_file on Flagged Spans"]
        end

        Chain --> MCP
        Direct --> MCP
        Direct --> CLI
        Direct -.-> Fit
        MCP --> Skip
        MCP --> Edit
        CLI --> Edit
    end
    classDef default fill:#ffffff,stroke:#475569,color:#0f172a,stroke-width:1px
    style Canvas fill:#ffffff,stroke:#cbd5e1,stroke-width:1.5px,color:#0f172a
    style Hosts fill:#f8fafc,stroke:#94a3b8,color:#0f172a
    style Paths fill:#f8fafc,stroke:#94a3b8,color:#0f172a
    style Actions fill:#f8fafc,stroke:#94a3b8,color:#0f172a
    style Fit fill:#ffffff,stroke:#94a3b8,stroke-dasharray: 5 5,color:#64748b
```

### Six Core Engineering Innovations in `v2.0.0`

1. **4-Backtick Fence-Safe Multi-Format Chunker (`Stage 0`)**:
   - Tracks 6-level CommonMark heading breadcrumbs (`H1 > H2 > ... > H6`) and outer 4-backtick/tilde fences (` ```` `) so nested 3-backtick code examples and Python `#` comments inside code blocks never fracture sections. Natively chunks `.md`, `.html`, `.txt`, `.json`, `.jsonl`, and `.csv` with disambiguated relative `#L<start>-L<end>` spans and script-aware token budgets (`≤ 380` English / `≤ 780` Multilingual tokens) so `ModernBERT`'s `512`-token window never silently clips tail claims.
2. **Head-Language Guard & Per-Chunk Bilingual Routing**:
   - Routes English chunks to `ModernBERT-large` (`subfolder=None`) and multilingual chunks to `mmBERT` (`subfolder="multilingual"`), protected by the **Head-Language Guard**: whenever shared `query`, `ssot`, `labels`, or `questions` criteria contain non-English characters (Thai, CJK, Devanagari, Cyrillic, Arabic), **100%** of chunks route to `multilingual` so English passages never cross-attend against out-of-vocabulary non-English head tokens.
3. **3-Question Joint NLI Contradiction Triad (`P_eff`)**:
   - Single-question contradiction prompts hallucinate conflicts on unrelated topics or verbatim SSOT restatements. `/xlaya` evaluates `P_eff = P(is_relevant) × P(contradicts_ssot) × (1 − P(is_consistent))` in a single batched pass, achieving **`8 / 8 (100%)`** counter-anchor accuracy with **`0` false positives** on verbatim, consistent, and off-topic passages.
4. **Offline-First Snapshot Loader & Length-Sorted Dynamic Batching (`7.92×`–`8.9×` Speedup)**:
   - `_resolve_local_laya_snapshot()` resolves local HuggingFace cache snapshots directly without blocking network round-trips, while `predict_batch(..., sort_by_length=True)` eliminates sequence-padding waste across heterogeneous chunks (`0 / 432` batch-vs-single label flips).
5. **Calibrated Tournament Elimination for `>20` Choice Labels**:
   - Automatically switches to `laya.predict_tournament(..., group_size=16)` whenever a `choice` taxonomy exceeds `20` labels, preserving full cross-attention precision across multi-round brackets without embedding-collapse degradation.
6. **Zero-Extra-Dependency Resident MCP Server + Conformal `vega-U` Stage 2**:
   - `xlaya_mcp_server.py` enforces strict JSON-RPC 2.0 `stdio` wire purity (`sys.stdout` redirected to `sys.stderr`), keeps `en`, `multi`, and `vega-U` (`0.8b` / `4b` / `gemma-1b`, up to `73,728` tokens) warm at **`~30 ms`**, strips raw `unbound` / `action` fields to enforce conformal `prediction_set` + `abstain` + `route` gating, and auto-reclaims idle GPU VRAM after `LAYA_IDLE_TTL_SEC` (`1800 s`).

---

## 2. Empirical 4-Axis Benchmarks (`RTX 5060 Ti 16 GB VRAM`)

Full methodology and extended telemetry are documented in [`benchmarks/XLAYA_V2_BENCHMARK_REPORT.md`](./benchmarks/XLAYA_V2_BENCHMARK_REPORT.md) and [`benchmarks/xlaya_v2_benchmark_results.json`](./benchmarks/xlaya_v2_benchmark_results.json) (reproducible via [`benchmarks/run_xlaya_benchmark.py`](./benchmarks/run_xlaya_benchmark.py)).

### Axis 1: Cold vs. Resident Warm MCP & Official `predict_batch` Throughput

| Execution Path / Model (`RTX 5060 Ti 16 GB`) | Batch / Workload | Batched / v2 Latency ↓ | Baseline / Loop ↓ | Speedup ↑ | Throughput ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cold CLI Sweep (`laya_triage.py`)** | `32` chunks (`batch_size=32`) | **`9.41 s`** (`8,459.7 ms` engine) | `74.51 s` (`v1.0` CLI) | **`7.92×`** ✨ | `3.40 chunks/s` |
| **Resident Warm MCP (`xlaya_mcp_server.py`)** | `1` state / `32`-chunk Triad (`96` Qs) | **`29.99 ms` / `662.49 ms`** | `74,510 ms` (`v1.0` cold) | **`112.5×`** ✨ | `144.9 q/s` (`6.90 ms/q`) |
| **Stage 0 Pure-Python Sieve** | `27` specs / `1,165` chunks | **`170.29 ms`** (`0.15 ms/chunk`) | `173,666` full tokens | **`152.7×` tokens** | **`6,841.3 chunks/s`** |
| **`english` (`ModernBERT-large`, `4` Q/state)** | `1` / `8` / `64` states | **`26.4` / `97.0` / `853.8 ms`** | `39.5` (T4) / `263.8` / `2,119.4 ms` | **`2.7×–2.8×`** | **`329.8 q/s`** (`3.03 ms/q`) |
| **`multilingual` (`mmBERT-base`, `4` Q/state)** | `1` / `8` / `16` states | **`20.7` / `37.2` / `80.1 ms`** | `32.8` (T4) / `218.5` / `522.4 ms` | **`5.9×–6.5×`** ✨ | **`859.1 q/s`** (`1.16 ms/q`) ✨ |
| **`Router.predict_batch` (`50% EN / 50% HI`, `3` Q/s)** | `32` / `64` states | **`114.6 ms` / `236.4 ms`** | `1,016.2 ms` / `1,950.6 ms` | **`8.9×` / `8.3×`** ✨ | **`837.9 q/s`** (`279.3 states/s`) |

<details>
<summary><b>LaTeX (booktabs) Source for Academic Citations — Axis 1 Latency & Batch Throughput</b></summary>

```latex
\begin{table}[t]
\centering
\caption{Empirical latency and batched throughput of \textsc{XLaya v2.0.0} (\texttt{laya 0.4.31}, \texttt{vegaml 0.16.1}) on an NVIDIA GeForce RTX 5060 Ti (16\,GB VRAM, \texttt{PyTorch 2.13.0+cu130}). All forward passes emit zero autoregressive tokens (\texttt{output\_tokens = 0}).}
\label{tab:xlaya_axis1_throughput}
\begin{tabular}{llrrrr}
\toprule
\textbf{Engine / Backbone} & \textbf{Workload} & \textbf{Batched (ms)} $\downarrow$ & \textbf{Baseline / Loop (ms)} $\downarrow$ & \textbf{Speedup} $\uparrow$ & \textbf{Throughput} $\uparrow$ \\
\midrule
Cold CLI (\texttt{laya\_triage.py}) & 32 chunks & \textbf{9,410.0} & 74,510.0 (\texttt{v1.0}) & \textbf{7.92$\times$} & 3.4 chunks/s \\
Warm Resident MCP (\texttt{xlaya\_predict}) & 1 state / 32-chunk Triad & \textbf{29.99 / 662.49} & 74,510.0 (\texttt{v1.0}) & \textbf{112.5$\times$} & 144.9 q/s \\
Stage 0 Pure-Python Sieve & 27 specs (1,165 chunks) & \textbf{170.29} & --- & 152.7$\times$ tok & \textbf{6,841.3 chunks/s} \\
\texttt{ModernBERT-large} (\texttt{english}) & $B=8 \times 4$ questions & \textbf{97.0} & 263.8 (loop) & 2.7$\times$ & 329.8 q/s \\
\texttt{mmBERT-base} (\texttt{multilingual}) & $B=8 \times 4$ questions & \textbf{37.2} & 218.5 (loop) & 5.9$\times$ (6.5$\times$ at $B=16$) & \textbf{859.1 q/s} \\
\texttt{Router.predict\_batch} (50\% EN / 50\% HI) & $B=32 \times 3$ questions & \textbf{114.6} & 1,016.2 (loop) & \textbf{8.9$\times$} & \textbf{837.9 q/s} \\
\bottomrule
\end{tabular}
\end{table}
```

</details>

---

### Axis 2: Batch-vs-Single Numerical Determinism (`0 / 432` Flips), Cross-Lingual Router & Presets

- **Zero Discrete Label Flips (`0 / 432`)**: Across `144` `english` (`max_abs_dp = 0.00580`), `144` `multilingual` (`max_abs_dp = 0.00260`), and `144` `Router.predict_batch` (`max_abs_dp = 0.00650`) paired evaluations, dynamic length-sorted batching produced **`0` label or routing flips**.
- **`5 / 5 (100%)` Cross-Lingual Routing & 5 Built-In Presets**: Verified `100%` language routing accuracy across `EN`, `DE`, `HI`, `JA`, and `TH`, plus sub-50 ms multi-question preset execution:

| Preset / Check | Latency ↓ | Questions / Scope | Verified Telemetry Receipt |
| :--- | :---: | :---: | :--- |
| **Cross-Lingual Router (`EN/DE/HI/JA/TH`)** | `20.7–26.4 ms` | `5` languages | **`5 / 5 (100%)`** route accuracy + Head-Language Guard protection ✓ |
| **`moderation` / `email` Presets** | **`27.10 ms` / `32.71 ms`** | `5` questions each | `toxic=0.0` (clean) • `is_phishing=0.984`, `is_spam=0.993` ✓ |
| **`guard` / `router` / `triage` Presets** | `40.74` / `49.56` / `49.80 ms` | `4–5` questions each | `prompt_injection=1.000` • `domain="math_or_logic"` • `intent="cancellation"` ✓ |

---

### Axis 3: Conformal Calibration, Joint NLI Contradiction Triad & XERJ `#1108` Watcher Gate

| Benchmark Suite & Evaluation Target | Baseline / Cold Reference | `/xlaya v2.0.0` Calibrated Result | Empirical Gain & Safety Receipt |
| :--- | :--- | :--- | :--- |
| **1. `routing_labels.csv` (`40` Fit / `20` Test)**<br>*(Conformal Few-Shot Calibration)* | Cold `vega-U 0.8B`: `60.0%` acc (`auto: 8, review: 12`)<br>`laya-typed-decisions`: `75.0%` (`25.75 ms`, `ECE 0.188`)<br>`laya` `english`: `80.0%` (`31.73 ms`, `ECE 0.283`) | **`vega-U 0.8B` Fitted (`40` rows in `7.58 s`)**:<br>**`90.0%` (`18/20`) team acc**, `ECE = 0.214`, `urgency_mae = 0.650` (`103.36 ms`) | **`+30.0 pts` over cold `vega-U`** & **`+10.0 pts` over zero-shot `laya`** in `< 8 s` fit time ✨ |
| **2. 3-Question Joint NLI Triad (`P_eff`)**<br>*(Counter-Anchor Contradiction Suite)* | 1-question `noul` suffers off-topic & verbatim-agreement false positives | **`8 / 8 (100.0%)` accuracy**, **`0` false positives**<br>`direct=0.6735`, `numeric=0.6806`, `ood_pgbouncer=0.7553` vs `0.0456` consistent, `73k_vega=0.6569` vs `0.0412` | **`16.6×` OOD conflict-vs-consistent separation** + `73,728`-token whole-file support ✨ |
| **3. `xerj-org/xerj#1108` Watcher Gate**<br>*(`120` Emails, Wrong-and-Confident `p ≥ 0.80`)* | Uncalibrated Baseline: **`24 / 98` (`24.49%` FAIL)** wrong-and-confident fires | Standalone Gate: **`0 / 30 (0.00% PASS)`**<br>Agreement Gate: `2 / 57` (`3.51% PASS`)<br>Ensemble: **`0 / 36 (0.00% PASS, 96/120 = 80.0% acc)`** | **`-100%` elimination of wrong-and-confident fires** (`24.49%` → `0.00%`) ✨ |

<details>
<summary><b>LaTeX (booktabs) Source for Academic Citations — Axis 3 Conformal Calibration & Contradiction Gates</b></summary>

```latex
\begin{table}[t]
\centering
\caption{Conformal calibration on \texttt{routing\_labels.csv} ($N_{\text{fit}}=40$, $N_{\text{test}}=20$) and boundary-gate verification on \texttt{xerj-org/xerj\#1108} ($N=120$ emails, confident threshold $p \ge 0.80$).}
\label{tab:xlaya_axis3_calibration}
\begin{tabular}{llrrrr}
\toprule
\textbf{Suite} & \textbf{Configuration} & \textbf{Accuracy} $\uparrow$ & \textbf{ECE / Rate} $\downarrow$ & \textbf{Urg.\ MAE} $\downarrow$ & \textbf{Median (ms)} $\downarrow$ \\
\midrule
\texttt{routing\_labels} & \texttt{laya-typed-decisions} (0-shot) & 75.0\% (15/20) & \textbf{0.188} & 0.671 & \textbf{25.75} \\
\texttt{routing\_labels} & \texttt{laya} \texttt{english} (0-shot) & 80.0\% (16/20) & 0.283 & 0.780 & 31.73 \\
\texttt{routing\_labels} & \texttt{vega-U 0.8B} (Cold 0-shot) & 60.0\% (12/20) & 0.222 & \textbf{0.650} & 64.36 \\
\texttt{routing\_labels} & \textbf{\texttt{vega-U 0.8B} (Fitted 40 rows, 7.58\,s)} & \textbf{90.0\% (18/20)} & 0.214 & \textbf{0.650} & 103.36 \\
\midrule
XERJ \texttt{\#1108} & Uncalibrated Baseline (FAIL) & --- & 24.49\% (24/98) & --- & --- \\
XERJ \texttt{\#1108} & \textbf{\textsc{XLaya} Standalone / Ensemble (PASS)} & \textbf{80.0\% (96/120)} & \textbf{0.00\% (0/30, 0/36)} & --- & \textbf{29.99} \\
\bottomrule
\end{tabular}
\end{table}
```

</details>

---

## 3. Resident MCP Server (`5` Tools) & Standalone CLI Usage

### A. Optional Neural Dependencies (`Stage 0` Runs on Python 3.9+ Stdlib Alone)
```bash
# Stage 1 (Laya ModernBERT-large + mmBERT) & Stage 2 (VegaML conformal vega-U):
pip install "laya>=0.4.31" "vegaml>=0.16.1" torch
```

### B. Register the Resident MCP Server (`xlaya_mcp_server.py` — `~30 ms` Warm)
Add `xlaya` to your MCP client configuration (`mcp_config.json`, Claude Desktop, Cursor, Antigravity, Windsurf, or Cline):
```json
{
  "mcpServers": {
    "xlaya": {
      "command": "python",
      "args": ["/path/to/xlaya/xlaya_mcp_server.py"],
      "env": {
        "LAYA_IDLE_TTL_SEC": "1800"
      }
    }
  }
}
```

| Resident MCP Tool | Core Parameters | Primary Use Case |
| :--- | :--- | :--- |
| **`xlaya_triage`** | `paths`, `mode` (`shortlist\|contradiction\|rubric\|classify`), `query`, `ssot`, `vega`, `lexical_prefilter` | End-to-end Stage 0 → Stage 1 → Stage 2 directory/file sweep returning `#L<start>-L<end>` spans |
| **`xlaya_predict`** | `questions`, `state` or `states`, `model` (`auto\|en\|multi`), `vega`, `vega_model` (`0.8b\|4b\|gemma-1b`) | In-memory `choice`, `score`, and `noul` evaluation (auto-runs Head-Language Guard & `predict_tournament`) |
| **`xlaya_status`** | *(none)* | Inspects warm-loaded models (`en`, `multi`, `vegaml`), active GPU/MPS/CPU device, and `LAYA_IDLE_TTL_SEC` |
| **`xlaya_preload`** | `models=["en", "multi"]`, `vega=false`, `vega_model="0.8b"`, `device="auto"` | Pre-warms checkpoints in RAM/VRAM ahead of latency-critical sweeps |
| **`xlaya_unload`** | *(none)* | Immediately evicts cached models and reclaims CPU RAM and CUDA VRAM |

### C. Run as a Standalone Python CLI (`laya_triage.py`)
```bash
# 1. Hybrid Section Shortlisting (#L<start>-L<end>) with Stage 0 lexical prefilter:
python laya_triage.py ./docs --mode shortlist --query "conformal prediction routing" --top-k 8 --lexical-prefilter 64

# 2. 3-Question Joint NLI Contradiction Sweep across Markdown, HTML, JSON, JSONL, and CSV:
python laya_triage.py . --mode contradiction --ssot "ModernBERT-large uses a 512-token context window and outputs 0 LLM tokens." --threshold 0.60

# 3. Completeness Rubric (0..4) + Stage 2 vega-U (0.8b) Conformal Escalation on Top-10 Finalists:
python laya_triage.py ./skills --mode rubric --ssot "Production specification with CLI and MCP schemas" --vega --vega-model 0.8b --vega-top-k 10

# 4. Whole-File Long-Document Conformal Check (up to 73,728 tokens) & Reproducible Self-Tests:
python laya_triage.py README.md --mode contradiction --ssot "XLaya v2.0.0 runs with zero output tokens." --whole-file --vega
python laya_triage.py --self-test
python benchmarks/run_xlaya_benchmark.py --mode quick
```

---

## 4. Universal Installation & Repository Layout

Install `/xlaya` into all detected agent skill directories on **Windows, macOS, or Linux** (automatically removing legacy `/layax` or `/laya` folders if present):

```powershell
# Windows PowerShell (installs into ~/.gemini/config/skills, ~/.claude/skills, ~/.codex/skills, ~/.agents/skills):
.\scripts\install.ps1 -All
```

```bash
# macOS / Linux (copy or --symlink into all detected agent skill roots, or --target <custom_dir>):
chmod +x scripts/install.sh
./scripts/install.sh --all
```

| Agent Class / Host | Target Skill Directory | Activation Command | Compatibility |
| :--- | :--- | :--- | :--- |
| **Any AI Agent (`Cloud • IDE • CLI • Local LLM`)** | `~/.*/skills/xlaya` or `<repo>/.agents/skills/xlaya` | `/xlaya` | `universal` (zero-config `<SKILL_DIR>` resolution + pure-Python fallback) |
| **Google Gemini / Antigravity (`IDE` & `agy` CLI)** | `~/.gemini/config/skills/xlaya` | `/xlaya` (chains with `/xsync`, `/xbrief`, `/xplan`, `/xdeep-research`, `/xpublish`, `/xultra`, `/xlearn`, `/xocr`) | `tested` (Windows 11 + RTX 5060 Ti 16 GB + Resident MCP) |
| **Claude Code / Desktop** | `~/.claude/skills/xlaya` | `/xlaya` | `spec-compatible` (`SKILL.md` + `stdio` MCP server + CLI) |
| **Grok / Cursor / Codex / Windsurf / Cline / OpenHands** | `~/.grok/skills/xlaya`, `~/.codex/skills/xlaya`, or `~/.agents/skills/xlaya` | `/xlaya` | `spec-compatible` (`SKILL.md` + `stdio` MCP server + CLI) |

### Repository Layout
```text
xlaya/
├── README.md                                  # Flagship architecture, 4-axis benchmarks & quick start
├── LICENSE                                    # MIT License + ConvAI Innovations upstream notice
├── llms.txt                                   # Concise machine-readable entrypoint index for LLMs
├── laya_triage.py                             # Standalone 3-stage CLI & programmatic Python engine
├── xlaya_mcp_server.py                        # Zero-extra-dependency JSON-RPC 2.0 stdio MCP server
├── scripts/
│   ├── install.ps1                            # Windows PowerShell multi-agent installer
│   └── install.sh                             # macOS / Linux POSIX bash installer (LF normalized)
├── skills/
│   └── xlaya/
│       ├── SKILL.md                           # Portable /xlaya agent skill specification
│       └── scripts/                           # SHA-256 mirrored CLI & MCP server for standalone skill copy
│           ├── laya_triage.py
│           └── xlaya_mcp_server.py
└── benchmarks/
    ├── XLAYA_V2_BENCHMARK_REPORT.md           # Full 4-axis RTX 5060 Ti technical benchmark report
    ├── xlaya_v2_benchmark_results.json        # Self-documented JSON telemetry receipts
    └── run_xlaya_benchmark.py                 # Reproducible --mode {quick,full} verification harness
```

---

## 5. Related Toolkits & Author

- **[`iscvlab/xboost`](https://github.com/iscvlab/xboost)**: Composable 15-skill `/x*` AI agent engineering ecosystem, 3-Tier Compound Learning Playbooks, and 10 Python CLI & MCP verification engines (`10,382` LOC).
- **[`iscvlab/xocr`](https://github.com/iscvlab/xocr)**: High-speed Hybrid Document AI, Two-Stage LaTeX `booktabs` & HTML Table Surgery, Thai PUA/Sara-Am normalizer, and local `Typhoon-OCR v1.5` × `Baidu Unlimited-OCR-2` engine.
- **[`iscvlab/mcp-bridges`](https://github.com/iscvlab/mcp-bridges)**: Model Context Protocol (`MCP`) automation bridges for QGIS, Rhino 3D, Grasshopper, Adobe Illustrator, Figma, and Ghost Mouse.
- **[`iscvlab/sse-space-syntax`](https://github.com/iscvlab/sse-space-syntax)**: High-performance C++/Rust/Python Space Syntax, Axial/Segment Angular Choice, VGA, and Isovist simulation engine.

Engineered and maintained by **[Napat (Tan) Phasundhiae (`@napattan`)](https://github.com/napattan)** at **[ISCV | Isocurve Research Lab (`@iscvlab`)](https://github.com/iscvlab)**. Released under the **[MIT License](./LICENSE)**; Stage 1 and Stage 2 neural backends interoperate with ConvAI Innovations' open-source **[`laya`](https://github.com/NandhaKishorM/laya)** and `vegaml` libraries.
