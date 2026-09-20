# Sharing KV Caches Across Fine-Tuned Language Models with a Head-Wise Linear Map

[![DOI](https://img.shields.io/badge/DOI-Zenodo-024dad.svg)](https://zenodo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official implementation and empirical evaluation suite for the research paper:  
**"Sharing KV Caches Across Fine-Tuned Language Models with a Head-Wise Linear Map"** by *Mohammad Shahid Chaudhary*.

---

## Table of Contents
1. [Core Theoretical Concept](#core-theoretical-concept)
2. [Key Empirical Results](#key-empirical-results)
3. [Environment Setup & Installation](#environment-setup--installation)
4. [Using `research.py` to Reproduce Benchmarks](#using-researchpy-to-reproduce-benchmarks)
   - [Table I: Downstream Memorized Task Retention (ARC-Easy)](#table-i-downstream-memorized-task-retention-arc-easy)
   - [Table II: Micro-Kernel Latency & Memory Scaling (128 to 16,384 tokens)](#table-ii-micro-kernel-latency--memory-scaling-128-to-16384-tokens)
   - [Table III: Full-Vocabulary Generative Fidelity (WikiText-2)](#table-iii-full-vocabulary-generative-fidelity-wikitext-2)
   - [Table IV: Cross-Specialist Generative Transfer & Free Autoregressive Rollout](#table-iv-cross-specialist-generative-transfer--free-autoregressive-rollout)
   - [End-to-End Execution (All 4 Tables)](#end-to-end-execution-all-4-tables)
5. [Complete CLI Argument Reference](#complete-cli-argument-reference)
6. [Hardware & Execution Tips (Kaggle / Colab / Cloud GPUs)](#hardware--execution-tips)
7. [Citation](#citation)

---

## Core Theoretical Concept

When serving multiple fine-tuned variants of a shared base model (e.g., in multi-agent routing or pipeline handoffs), each model redundantly pre-fills and allocates dedicated Key-Value (KV) memory for identical prompt prefixes. At 16-bit precision, a 16k-token context in GPT-2 consumes $\approx 576\text{ MB}$ per model instance.

Directly reusing another model's cache fails because fine-tuning induces **representational drift** in the key and value geometry. This codebase demonstrates that at GPT-2 scale, this drift is largely **linear**:

1. **Head-Wise Block-Diagonal Linear Map ($W_{\mathrm{map}}$):**  
   A parameter-efficient adapter consisting of two bias-free matrices per head per layer:
   $$K_B \approx K_A W_k, \qquad V_B \approx V_A W_v, \qquad W_k, W_v \in \mathbb{R}^{d \times d}$$
   For GPT-2 small ($L=12, H=12, d=64$), this totals only $1.20\text{M}$ parameters ($0.97\%$ of the 124M backbone).

2. **Constant-Time $O(1)$ Dual-Projection Invariant:**  
   Applying $W_k$ and $W_v$ to all $S$ cached tokens at each decoding step would cost $O(S d^2)$, eliminating the latency benefit of sharing. Because $W_{\mathrm{map}}$ is linear, bias-free, and confined to individual attention heads, associativity allows the projection to fold into the incoming query and attention output:
   $$\mathrm{Attn}(Q,\, K W_k,\, V W_v) = \mathrm{softmax}\!\left(\frac{(Q W_k^\top) K^\top}{\sqrt{d}}\right) V \, W_v = \mathrm{Attn}(Q W_k^\top,\, K,\, V) W_v$$
   - **Historical cache in memory is never modified or rewritten.**
   - **Per-step conversion cost is reduced from $O(S)$ to strictly $O(1)$ in context length $S$.**
   - **Numerical identity is exact down to machine precision ($\max |Q W_k^\top - (K W_k)^\top| = 0.00 \times 10^0$).**

3. **Multi-Objective Distillation:**  
   Adapters are optimized with frozen base and specialist backbones using joint output-space Kullback-Leibler divergence ($T = 2.0$) and intermediate geometric alignment ($\lambda = 0.5$):
   $$\mathcal{L} = T^2 D_{\mathrm{KL}}(P_B \,\|\, P_{\hat{B}}) + \frac{\lambda}{L}\sum_{l=1}^{L} \Big[ \mathrm{MSE}(\hat{K}^{(l)}, K_B^{(l)}) + \mathrm{MSE}(\hat{V}^{(l)}, V_B^{(l)}) + 2\big(d_{\cos}(\hat{K}^{(l)}, K_B^{(l)}) + d_{\cos}(\hat{V}^{(l)}, V_B^{(l)})\big)\Big]$$

---

## Key Empirical Results

- **100% Marginal KV Cache Memory Saved:** The consumer specialist stores zero cache entries of its own.
- **Table I (Memorized Retention):** 99.90% accuracy on ARC-Easy (matching the unshared teacher exactly).
- **Table II (Decode Scaling):** Up to $3.8\times$ micro-kernel speedup over naive cache transformation at $S = 16{,}384$ tokens.
- **Table III (Generative Fidelity):** 93.35% Top-1 agreement, 99.77% Confident Top-1 agreement, and $31.30$ PPL (vs. $31.10$ native teacher) over 64,000 evaluated tokens on WikiText-2.
- **Table IV (Peer-to-Peer Rollout):** Under free 32-token autoregressive generation (zero teacher-forcing), unmapped raw transfer collapses (1.0% exact match, 4.6 token survival), whereas $W_{\mathrm{map}}$ extends survival to 13.1 tokens and increases sequence exact match up to $11\times$.

---

## Environment Setup & Installation

### 1. Prerequisites
- Python 3.9 or newer
- NVIDIA GPU with CUDA support recommended (CPU execution is supported automatically)

### 2. Install Dependencies
```bash
pip install torch transformers datasets tqdm
```

*(Optional: for enhanced attention kernels or profiling)*
```bash
pip install accelerate
```

---

## Using `research.py` to Reproduce Benchmarks

The script `research.py` (or `paper/research.py`) is fully self-contained. It handles downloading/caching datasets, fine-tuning specialist teacher models, distilling $W_{\mathrm{map}}$ adapters, and compiling the publication benchmark tables.

### Table I: Downstream Memorized Task Retention (ARC-Easy)
Evaluates whether a fine-tuned specialist can retrieve memorized answers when reading a cache produced entirely by the unadapted base GPT-2 model:
```bash
python research.py --only_table 1 --num_samples 1000 --teacher_epochs 10 --epochs 15
```
- **What it does:**
  1. Fine-tunes an ARC-Easy memorization teacher on 1,000 science QA samples (cached to `./gpt2_memorized_arc`).
  2. Distills $W_{\mathrm{map}}$ using output-space KL divergence and activation alignment.
  3. Evaluates downstream choice accuracy for:
     - Base GPT-2 (Zero-shot baseline: $\sim 26.2\%$)
     - Fine-tuned Teacher (Upper bound: $99.90\%$)
     - SharedKV Pure Single $W_{\mathrm{map}}$ ($99.90\%$, with 100% specialist KV memory saved)

---

### Table II: Micro-Kernel Latency & Memory Scaling (128 to 16,384 tokens)
Benchmarks per-step decode latency across 12 layers for context lengths $S \in \{128, 512, 1024, 2048, 4096, 8192, 16384\}$:
```bash
python research.py --only_table 2
```
- **What it does:**
  1. Numerically validates the query-space dual projection identity:
     $$\max |Q W_k^\top - (K W_k)^\top| = 0.00 \times 10^0$$
  2. Compares 12-layer decoding step latency across three modes:
     - **Native Attention:** Unshared attention baseline.
     - **Naive Cache Transform:** Rewriting $K W_k$ and $V W_v$ over all $S$ tokens (scales linearly with $S$).
     - **Dual-Projection Kernel:** Transforming $Q W_k^\top$ and $O W_v$ on the 1-token query (strictly $O(1)$).
  3. Reports analytical fp16 memory footprint and speedup factors (up to $3.8\times$ at $S=16384$).

---

### Table III: Full-Vocabulary Generative Fidelity (WikiText-2)
Measures generative fidelity and perplexity over 64,000 held-out test tokens (1,000 sequences $\times$ 64 continuation tokens) across the entire 50,257-token vocabulary:
```bash
python research.py --only_table 3 --eval_wikitext_chunks 1000 --hybrid_ratio 0.20
```
- **What it does:**
  1. Fine-tunes a causal language modeling specialist on the WikiText-2 training split (cached to `./gpt2_teacher_wikitext`).
  2. Distills $W_{\mathrm{map}}$ over multi-token sequences.
  3. Evaluates held-out perplexity, Top-1 agreement, Top-2 agreement, Confident Top-1 (logit margin $\ge 1.0$), Top-5 overlap, logit cosine similarity, and KL divergence ($T=2$).
  4. Evaluates the **Saliency-Guided Mixed Cache** frontier: substituting the top 20% attention-salient drifting tokens with native cache while sharing the remaining 80%.

---

### Table IV: Cross-Specialist Generative Transfer & Free Autoregressive Rollout
Tests peer-to-peer cache transfer between two independently fine-tuned domain specialists (neither being the base model):
- **Specialist A:** MathQA (trained on GSM8K)
- **Specialist B:** PythonCode (trained on Alpaca Python instructions)
```bash
python research.py --only_table 4 --domain_adapter_epochs 5 --rollout_len 32
```
- **What it does:**
  1. Fine-tunes both specialists on disjoint domain splits (cached to `./gpt2_specialists/`).
  2. Trains directional linear adapters:
     - $\text{MathQA} \rightarrow \text{PythonCode}$
     - $\text{PythonCode} \rightarrow \text{MathQA}$
  3. **Teacher-Forced Evaluation:** Computes Native PPL, Raw PPL, $W_{\mathrm{map}}$ PPL, Raw Top-1, and $W_{\mathrm{map}}$ Top-1 over held-out test chunks.
  4. **Free Autoregressive Rollout (Zero Teacher Forcing):** Generates 32 greedy tokens conditioned on the source cache. Evaluates:
     - **Sequence Exact Match (Seq EM):** Percentage of sequences where all 32 tokens match the native trajectory.
     - **Survival Length:** Mean generation step of the first token divergence (out of 32).

---

### End-to-End Execution (All 4 Tables)
To run the full paper evaluation pipeline sequentially:
```bash
python research.py
```
*Note: Checkpoints for teachers are automatically cached in local directories (`./gpt2_memorized_arc`, `./gpt2_teacher_wikitext`, `./gpt2_specialists`). Subsequent runs will reuse these checkpoints.*

---

## Complete CLI Argument Reference

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--base_model` | `str` | `"gpt2"` | Base HuggingFace checkpoint name or path. |
| `--only_table` | `int` | `0` | Run only a specific table (`1`, `2`, `3`, or `4`). `0` executes all tables. |
| `--skip_table1` | `flag` | `False` | Skip Table I (ARC-Easy downstream memorization). |
| `--skip_table2` | `flag` | `False` | Skip Table II (Micro-kernel latency & memory scaling). |
| `--skip_table3` | `flag` | `False` | Skip Table III (WikiText-2 full-vocabulary generative fidelity). |
| `--skip_table4` | `flag` | `False` | Skip Table IV (Cross-specialist transfer & free rollout). |
| `--force_retrain_teacher`| `flag` | `False` | Force retraining teacher models from scratch, ignoring cached checkpoints. |
| `--num_samples` | `int` | `1000` | Sample count for Table I ARC-Easy evaluation. |
| `--teacher_epochs` | `int` | `10` | Supervised fine-tuning epochs for ARC-Easy teacher specialization. |
| `--teacher_lr` | `float` | `4e-4` | Learning rate for downstream teacher training with AdamW. |
| `--epochs` | `int` | `15` | Distillation epochs for Table I $W_{\mathrm{map}}$ adapter. |
| `--wikitext_chunks` | `int` | `1000` | Sequence chunks for WikiText-2 teacher fine-tuning. |
| `--wikitext_epochs` | `int` | `3` | Epochs for WikiText-2 teacher fine-tuning. |
| `--wikitext_adapter_chunks`| `int` | `1000`| Training chunks for WikiText-2 adapter distillation. |
| `--wikitext_adapter_epochs`| `int` | `5` | Epochs for WikiText-2 adapter distillation. |
| `--eval_wikitext_chunks` | `int` | `1000` | Evaluation chunks for Table III (1000 chunks $\times$ 64 tokens = 64,000 tokens). |
| `--domain_chunks` | `int` | `1000` | Training chunks for MathQA and PythonCode specialist fine-tuning. |
| `--domain_epochs` | `int` | `3` | Fine-tuning epochs for domain specialists. |
| `--domain_adapter_chunks`| `int` | `1000` | Training chunks for cross-specialist adapter distillation. |
| `--domain_adapter_epochs`| `int` | `5` | Epochs for cross-specialist adapter distillation (Table IV). |
| `--domain_eval_chunks` | `int` | `1000` | Evaluation chunks for Table IV cross-specialist transfer. |
| `--rollout_len` | `int` | `32` | Generation length for free autoregressive rollout stress test. |
| `--rollout_chunks` | `int` | `0` | Sequence count for free rollout evaluation (`0` evaluates all test chunks). |
| `--hybrid_ratio` | `float` | `0.20` | Fraction of tokens with highest drift replaced with native cache (default: 0.20 = 80% memory saved). |
| `--act_reg_weight` | `float` | `0.5` | Regularization weight $\lambda$ for intermediate key-value geometric alignment ($\text{MSE} + 2\cdot d_{\cos}$). |
| `--lr` | `float` | `1e-3` | Peak learning rate for $W_{\mathrm{map}}$ adapter with Cosine Annealing. |
| `--kd_temp` | `float` | `2.0` | Softmax temperature for predictive distribution matching ($T=2.0$). |
| `--seed` | `int` | `42` | Global deterministic random seed. |
| `--device` | `str` | `""` | Device override (`"cuda"`, `"cpu"`, or specific device `"cuda:0"`). |

---

## Hardware & Execution Tips

### 1. Kaggle / Google Colab (Single or 2x T4 GPU)
- **Single Cell Execution:** Run tables individually if execution time per cell is limited:
  ```bash
  # Step 1: Run Table 1 & Table 2
  python research.py --only_table 1
  python research.py --only_table 2

  # Step 2: Run Table 3 (WikiText-2)
  python research.py --only_table 3

  # Step 3: Run Table 4 (Cross-Specialist)
  python research.py --only_table 4
  ```
- **Dual GPU Notes:** The script defaults to device auto-selection (`cuda` if available). On multi-GPU systems, you can target specific devices using `--device cuda:0`.
- **Reproducibility:** All data partitions and weight initializations are pinned to seed `42` by default.

### 2. Disk Caching
- Fine-tuned specialist checkpoints are written to disk upon first completion:
  - `./gpt2_memorized_arc`
  - `./gpt2_teacher_wikitext`
  - `./gpt2_specialists`
- If you wish to retrain from scratch, add the `--force_retrain_teacher` flag.

---

## Citation

```bibtex
@misc{chaudhary2026sharing,
  title        = {Sharing KV Caches Across Fine-Tuned Language Models with a Head-Wise Linear Map},
  author       = {Chaudhary, Mohammad Shahid},
  year         = {2026},
  month        = {sep},
  note         = {Zenodo},
  doi          = {10.5281/zenodo.xxxxxxx},
  url          = {https://doi.org/10.5281/zenodo.xxxxxxx}
}
```
