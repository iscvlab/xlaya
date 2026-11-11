# Empirical 4-Axis Benchmark Report: `iscvlab/xlaya` (`v2.0.0`) × `Laya v0.4.31` × `VegaML v0.16.1`

> [!IMPORTANT]
> **Executive Summary**
> This technical report documents primary-source GPU and CPU benchmarks for **`iscvlab/xlaya` (`v2.0.0`)**, a non-autoregressive **3-Stage System 1 → System 1.5 Semantic Decision, Joint NLI Contradiction Triad & Conformal Triage Sidecar**. Evaluated on an **NVIDIA GeForce RTX 5060 Ti (`16 GB VRAM`, `PyTorch 2.13.0+cu130`, `laya 0.4.31`, `vegaml 0.16.1`)**, `xlaya v2.0.0` cuts cold 32-chunk CLI sweeps by **`7.92×` (`74.51 s` → `9.41 s`)**, reaches **`29.99 ms` warm single-state inference** via its resident MCP server (`xlaya_mcp_server.py`), accelerates mixed-language `Router.predict_batch` throughput by **`8.9×` (`837.9 q/s`)** and `mmBERT` batch throughput to **`859.1 q/s` (`1.16 ms/question`)** with **`0 / 432` batch-vs-single label flips**, boosts `vega-U (0.8b)` routing accuracy from **`60.0%` cold to `90.0%` (`18/20`)** after a **`7.58 s` 40-row conformal fit**, achieves **`8 / 8 (100.0%)`** counter-anchor precision on its 3-Question Joint NLI Contradiction Triad, and eliminates wrong-and-confident (`p ≥ 0.80`) fires on **[`xerj-org/xerj#1108`](https://github.com/xerj-org/xerj/issues/1108)** from **`24.49%` (`24/98` FAIL) down to `0.00%` (`0/30` standalone, `0/36` calibrated ensemble PASS)**.

---

## 1. Benchmark Hardware, Software Stack & Visual Summary

- **GPU Hardware**: `NVIDIA GeForce RTX 5060 Ti` (`16 GB GDDR7 VRAM`)
- **Runtime Stack**: `PyTorch 2.13.0+cu130`, `laya 0.4.31` (`convaiinnovations/laya` — `ModernBERT-large` & `mmBERT`), `vegaml 0.16.1` (`vega-U` `0.8b` / `4b` / `gemma-1b`)
- **Autoregressive Output Tokens**: `0` (`output_tokens: 0` across all encoder and conformal forward passes)
- **Raw Telemetry & Harness**: [`benchmarks/xlaya_v2_benchmark_results.json`](./xlaya_v2_benchmark_results.json) & [`benchmarks/run_xlaya_benchmark.py`](./run_xlaya_benchmark.py)

### 20-Block Visual Performance Summary (`RTX 5060 Ti 16 GB VRAM`)

| Evaluation Dimension | Configuration / Engine | Visual Comparison Bar | Measured Result & Delta vs. Baseline |
| :--- | :--- | :--- | :--- |
| **1. Cold CLI Boot Speed ↑**<br>*(32-Chunk Contradiction Sweep)* | **`xlaya v2.0.0` (`predict_batch` + Local Snap)**<br>`xlaya v1.0` Sequential Baseline | `████████████████████` **`7.92× Faster`**<br>`███░░░░░░░░░░░░░░░░░` `1.0× Baseline` | **`9.41 s` wall (`8,459.7 ms` engine)** ✨<br>`74.51 s` wall (`-87.4%` latency cut) |
| **2. Mixed-Lang Batch Speed ↑**<br>*(50% EN / 50% HI, `32` States × `3` Qs)* | **`Router.predict_batch` (`sort_by_length`)**<br>Sequential `predict()` Loop | `████████████████████` **`8.9× Faster`**<br>`██░░░░░░░░░░░░░░░░░░` `1.0× Loop` | **`114.6 ms` (`837.9 q/s`, `0` flips)** ✨<br>`1,016.2 ms` (`279.3 states/s` batched) |
| **3. Peak Encoder Throughput ↑**<br>*(`8` States × `4` Questions/State)* | **`mmBERT` Multilingual Batched**<br>`ModernBERT-large` English Batched<br>Sequential `mmBERT` Loop | `████████████████████` **`859.1 q/s (5.9×)`**<br>`████████░░░░░░░░░░░░` `329.8 q/s (2.7×)`<br>`███░░░░░░░░░░░░░░░░░` `146.5 q/s (1.0×)` | **`1.16 ms/question` (`37.2 ms` batch)** ✨<br>`3.03 ms/question` (`97.0 ms` vs `263.8 ms`)<br>`218.5 ms` (`6.5×` speedup at `B = 16`) |
| **4. Conformal Fit Accuracy ↑**<br>*(`routing_labels.csv` Held-Out `N = 20`)* | **`vega-U 0.8B` Fitted (`40` rows, `7.58 s`)**<br>`laya` English Zero-Shot<br>`laya-typed-decisions` Zero-Shot<br>`vega-U 0.8B` Cold Zero-Shot | `██████████████████░░` **`90.0% (18/20)`**<br>`████████████████░░░░` `80.0% (16/20)`<br>`███████████████░░░░░` `75.0% (15/20)`<br>`████████████░░░░░░░░` `60.0% (12/20)` | **`ECE = 0.214` (`103.36 ms` median)** ✨<br>`31.73 ms` median (`ECE = 0.283`)<br>`25.75 ms` median (`ECE = 0.188`)<br>`64.36 ms` (`auto: 8`, `review: 12`) |
| **5. Contradiction & Watcher Gate ↑**<br>*(NLI Triad + `xerj-org/xerj#1108`)* | **`/xlaya` 3-Q NLI Triad (`8/8` Probes)**<br>**`/xlaya` Watcher Gate (`100% - FalseConf`)**<br>XERJ `#1108` Baseline (`24/98` FAIL) | `████████████████████` **`100.0% (8/8)`**<br>`████████████████████` **`100.0% (0/30 False)`**<br>`███████████████░░░░░` `75.51% (24.49% FAIL)` | **`16.6×` OOD conflict separation (`0` FP)** ✨<br>**`0.00%` wrong-confident (`80.0%` ens. acc)** ✨<br>`24 / 98` (`24.49%`) wrong at `p ≥ 0.80` |
| **6. Pass-B Token Compression ↑**<br>*(`27` Specs / `1,165` Chunks)* | **`/xlaya` Stage 0 Top-5 Shortlist**<br>Unfiltered Full-Corpus Read | `████████████████████` **`99.35% Saved (152.7×)`**<br>`░░░░░░░░░░░░░░░░░░░░` `0.0% Saved (1.0×)` | **`1,137` tok in `170.29 ms` (`6,841/s`)** ✨<br>`173,666` tok across `27` specifications |

---

## 2. Axis 1: Latency, Warm Resident MCP & Official `predict_batch` Throughput

### 2.1 End-to-End Pipeline Tiers (`Cold CLI` vs. `Warm Resident MCP` vs. `Stage 0 Sieve`)

| Execution Path | Workload Profile | Measured Latency ↓ | Throughput / Per-Unit Speed ↑ | Key Engineering Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **1. `xlaya v1.0` Cold CLI Baseline** | `32` Markdown chunks (sequential) | `74,510.0 ms` (`74.51 s`) | `0.43 chunks/s` | Remote HF Hub metadata checks + 1-by-1 `predict()` loop |
| **2. `xlaya v2.0.0` Cold CLI (`laya_triage.py`)** | `32` Markdown chunks (`batch_size=32`) | **`9,410.0 ms` (`9.41 s`)** | **`3.40 chunks/s` (`7.92×` faster)** | Offline-first `_resolve_local_laya_snapshot()` + `sort_by_length=True` |
| **3. Warm Resident MCP (`xlaya_predict`)** | `1` state in-memory decision | **`29.99 ms`** | **`33.3 states/s`** | Persistent `_MODEL_CACHE` + `stdio` JSON-RPC 2.0 wire purity |
| **4. Warm Resident MCP (`xlaya_triage`)** | `32` chunks × `3`-Q Joint NLI Triad (`96` Qs) | **`662.49 ms`** | **`144.9 q/s` (`6.90 ms/q`)** | Batched `ModernBERT-large` cross-attention (`112.5×` faster than v1 cold) |
| **5. Warm Resident MCP (`vega: true`)** | `1` state conformal `vega-U (0.8b)` | **`176.18 ms`** | **`5.68 states/s`** | Calibrated `prediction_set`, `abstain`, and `route` (`auto` vs `review`) |
| **6. Stage 0 Pure-Python Sieve** | `27` specs / `1,165` chunks (`173,666` tok) | **`170.29 ms`** | **`6,841.3 chunks/s` (`0.15 ms/chunk`)** | 4-backtick fence-safe splitter + BM25 + code-span + Thai 3-gram (`1,137` tok top-5) |

### 2.2 Official `bench_predict_batch.py` GPU Scaling (`RTX 5060 Ti 16 GB` vs. `T4` Reference)

| Model & Backbone | Batch Size (`States`) | Questions / State | Batched Latency ↓ (`ms`) | Sequential Loop ↓ (`ms`) | Batch Speedup ↑ | Throughput ↑ (`q/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`english` (`ModernBERT-large`)** | `1` | `4` | **`26.4 ms`** (`6.60 ms/q`) | `39.5 ms` *(T4 ref)* | **`1.5×` vs T4** | `151.5 q/s` |
| **`english` (`ModernBERT-large`)** | `4` | `4` | **`55.4 ms`** | `153.2 ms` | **`2.8×`** | `288.8 q/s` |
| **`english` (`ModernBERT-large`)** | `8` | `4` | **`97.0 ms`** (`3.03 ms/q`) | `263.8 ms` | **`2.7×`** | **`329.8 q/s`** |
| **`english` (`ModernBERT-large`)** | `64` | `4` | **`853.8 ms`** | `2,119.4 ms` | **`2.5×`** | `299.8 q/s` |
| **`multilingual` (`mmBERT-base`)** | `1` | `4` | **`20.7 ms`** (`5.18 ms/q`) | `32.8 ms` *(T4 ref)* | **`1.6×` vs T4** | `193.2 q/s` |
| **`multilingual` (`mmBERT-base`)** | `8` | `4` | **`37.2 ms`** (`1.16 ms/q`) | `218.5 ms` | **`5.9×`** | **`859.1 q/s`** ✨ |
| **`multilingual` (`mmBERT-base`)** | `16` | `4` | **`80.1 ms`** | `522.4 ms` | **`6.5×`** ✨ | `798.9 q/s` |
| **`multilingual` (`mmBERT-base`)** | `64` | `4` | **`362.4 ms`** | `1,786.6 ms` | **`4.9×`** | `706.5 q/s` |
| **`Router.predict_batch` (`50% EN / 50% HI`)** | `32` | `3` | **`114.6 ms`** | `1,016.2 ms` | **`8.9×`** ✨ | **`837.9 q/s`** (`279.3 states/s`) |
| **`Router.predict_batch` (`50% EN / 50% HI`)** | `64` | `3` | **`236.4 ms`** | `1,950.6 ms` | **`8.3×`** | `812.1 q/s` (`270.7 states/s`) |

---

## 3. Axis 2: Batch-vs-Single Numerical Determinism (`432` Comparisons)

Dynamic length-sorted batching (`sort_by_length=True`) introduces minor floating-point reduction order differences under CUDA tensor cores. Across **`432` paired question evaluations**, `predict_batch()` achieved **`0` discrete label flips** relative to single-state `predict()`:

| Model / Router Path | Total Paired Decisions | Max Absolute Posterior Delta (`max_abs_dp`) ↓ | Discrete Label / Choice Flips ↓ | Verification Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **`english` (`ModernBERT-large`)** | `144` | `0.00580` | **`0 / 144`** | `PASS` ✓ |
| **`multilingual` (`mmBERT-base`)** | `144` | `0.00260` | **`0 / 144`** | `PASS` ✓ |
| **`Router.predict_batch` (Mixed EN/HI)** | `144` | `0.00650` | **`0 / 144`** | `PASS` ✓ |
| **Combined Total** | **`432`** | **`≤ 0.00650`** | **`0 / 432 (0.00%)`** | **`100% Deterministic Labels`** ✨ |

---

## 4. Axis 3: Conformal Calibration, Joint NLI Contradiction Triad & XERJ `#1108` Gate

### 4.1 Conformal Few-Shot Calibration on Official `routing_labels.csv` (`40` Fit / `20` Held-Out Test)

| Model Configuration | Fit Overhead | Held-Out Team Accuracy ↑ | Team ECE ↓ | Urgency MAE ↓ | Median Latency ↓ | Conformal Routing Telemetry |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`laya-typed-decisions` (Zero-Shot)** | `0.00 s` | `75.0%` (`15/20`) | **`0.188`** | `0.671` | **`25.75 ms`** | Fast System 1 baseline |
| **`laya` `english` (Zero-Shot)** | `0.00 s` | `80.0%` (`16/20`) | `0.283` | `0.780` | `31.73 ms` | Strong zero-shot encoder head |
| **`vega-U (0.8b)` Cold (Zero-Shot)** | `0.00 s` | `60.0%` (`12/20`) | `0.222` | **`0.650`** | `64.36 ms` | Safely abstains (`auto: 8`, `review: 12`, `abstain: 12`) |
| **`vega-U (0.8b)` Fitted (`40` Rows)** | **`7.58 s`** | **`90.0%` (`18/20`)** ✨ | `0.214` (`urg_ece=0.059`) | **`0.650`** | `103.36 ms` | **`+30.0 pts` over cold** (`abstain: 0`, `set_sizes: {2:1, 4:6, 5:13}`) |

### 4.2 3-Question Joint NLI Contradiction Triad (`P_eff`) Counter-Anchor Probes (`8 / 8 = 100%`)

Naive 1-question contradiction prompts suffer from off-topic triggers and verbatim-agreement false positives. `/xlaya` evaluates a 3-question joint NLI triad in a single batched pass:

- **Effective Contradiction Formula**: `P_eff = P(is_relevant) × P(contradicts_ssot) × (1 − P(is_consistent))`

| Counter-Anchor Probe | Ground Truth | Measured `P_eff` | Threshold (`0.55`) Decision | Outcome |
| :--- | :---: | :---: | :---: | :---: |
| **1. Direct Architectural Conflict** (`128`-token vs `512`-token SSOT) | `Contradiction` | **`0.6735`** | `FLAG` (`≥ 0.55`) | `True Positive` ✓ |
| **2. Subtle Numeric Swap** (Port / TTL / limit inversion) | `Contradiction` | **`0.6806`** | `FLAG` (`≥ 0.55`) | `True Positive` ✓ |
| **3. Out-of-Domain `PgBouncer` Pool Conflict** | `Contradiction` | **`0.7553`** | `FLAG` (`≥ 0.55`) | `True Positive` (`16.6×` separation) ✨ |
| **4. Out-of-Domain `PgBouncer` Consistent Spec** | `Consistent` | **`0.0456`** | `PASS` (`< 0.55`) | `True Negative` ✓ |
| **5. Whole-File `73k`-Token `vega-U` Conflict** (`--whole-file --vega`) | `Contradiction` | **`0.6569`** | `FLAG` (`≥ 0.55`) | `True Positive` ✓ |
| **6. Whole-File `73k`-Token `vega-U` Consistent** (`--whole-file --vega`) | `Consistent` | **`0.0412`** | `PASS` (`< 0.55`) | `True Negative` ✓ |
| **7. Verbatim / Paraphrased Canonical SSOT Agreement** | `Consistent` | **`< 0.0500`** | `PASS` (`< 0.55`) | `True Negative` (`0` FP) ✓ |
| **8. Unrelated Off-Topic Infrastructure Docs** (Dataproc / UIA) | `Irrelevant` | **`< 0.0500`** | `PASS` (`< 0.55`) | `True Negative` (`0` FP) ✓ |

### 4.3 `xerj-org/xerj#1108` Watcher Boundary Gate (`120` Emails)

Evaluated on the 120-email Watcher Boundary benchmark from **[`xerj-org/xerj` Issue #1108](https://github.com/xerj-org/xerj/issues/1108)** measuring wrong-and-confident fires (`p ≥ 0.80`):

| Gate Strategy | Confident Fires (`p ≥ 0.80`) | Wrong-and-Confident Fires ↓ | Wrong-Confident Rate ↓ | Overall Accuracy / Gate Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **1. Uncalibrated Baseline Classifier** | `98 / 120` | `24 / 98` | `24.49%` | `FAIL` (High silent misrouting risk) |
| **2. `/xlaya` Standalone Boundary Gate** | `30 / 120` | **`0 / 30`** | **`0.00%`** ✨ | **`PASS` (`100%` precision on auto-fires)** |
| **3. Cross-Model Agreement Gate** | `57 / 120` | `2 / 57` | `3.51%` | `PASS` (`< 5.0%` error ceiling) |
| **4. `/xlaya` Calibrated Ensemble Gate** | `36 / 120` | **`0 / 36`** | **`0.00%`** ✨ | **`PASS` (`96 / 120 = 80.0%` overall acc)** |

---

## 5. Axis 4: Cross-Lingual Router (`5 / 5`) & Built-In Decision Presets

### 5.1 Cross-Lingual Language Routing (`EN`, `DE`, `HI`, `JA`, `TH`)
- **Per-State Language Routing Accuracy**: **`5 / 5 (100.0%)`** — English inputs route to `ModernBERT-large` (`subfolder=None`), while German (`DE`), Hindi (`HI`), Japanese (`JA`), and Thai (`TH`) inputs route cleanly to `mmBERT` (`subfolder="multilingual"`).
- **Head-Language Guard Invariant**: Whenever shared `questions` instructions or `criteria` contain non-English scripts (such as Thai or Japanese rubric heads), `/xlaya` forces **`100%` multilingual routing** across all chunks so English passages never cross-attend against out-of-vocabulary non-English head tokens.

### 5.2 Built-In Multi-Question Decision Presets (`RTX 5060 Ti` Warm Latency)

| Preset Name | Questions / Pass | Warm Latency ↓ (`ms`) | Verified Structured Output Sample |
| :--- | :---: | :---: | :--- |
| **`triage`** | `5` | `49.80 ms` | `intent="cancellation"`, `frustration=1.808`, `churn_risk=0.790`, `is_urgent=0.342` |
| **`email`** | `5` | `32.71 ms` | `category="security"`, `is_spam=0.993`, `is_phishing=0.984`, `urgency=1.505` |
| **`moderation`** | `5` | **`27.10 ms`** | `toxic=0.0`, `harassment=0.0`, `threat=0.0`, `spam=0.0`, `severity=0.340` |
| **`guard`** | `5` | `40.74 ms` | `prompt_injection=1.000`, `jailbreak=0.956`, `harm_severity=2.032`, `topic="coding"` |
| **`router`** | `4` | `49.56 ms` | `domain="math_or_logic"`, `difficulty=2.078`, `needs_tools=0.089`, `is_sensitive=0.0` |

---

## 6. Reproducing the Benchmarks Locally

```bash
# 1. Fast (< 15s) Stage 0 + Head-Language Guard + Warm Stage 1 Verification:
python benchmarks/run_xlaya_benchmark.py --mode quick

# 2. Full Stage 0 + Batched Stage 1 + Conformal Stage 2 (vega-U 0.8b) Suite:
python benchmarks/run_xlaya_benchmark.py --mode full

# 3. Built-in 10-Check Deterministic + Counter-Anchor + 24-Label Tournament Self-Test:
python laya_triage.py --self-test
```
