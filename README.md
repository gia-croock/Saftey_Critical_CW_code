This repo is adapted from the [DiffuGuard](https://github.com/yjydya/diffuguard) codebase.

Two attacks are evaluated (**DIJA** and **Context Nesting**) across a range of remasking-based defence configurations. Results are measured using ASR, perplexity, and structural adherence rate.

---

## Models Required

| Model | Purpose | Script |
|---|---|---|
| `GSAI-ML/LLaDA-8B-Instruct` | Victim model | `hf_models/model_download.sh` |
| `cais/HarmBench-Llama-2-13b-cls` | ASR_e evaluator | `download_harmbench_evaluator_model.sh` |
| `meta-llama/Llama-3.1-8B` | Perplexity evaluator | `download_llama3.8_8b.sh` |
| `Qwen2.5-7B-Instruct` | Prompt refiner (DIJA) | `hf_models/model_download.sh` |

> **Note:** `meta-llama/Llama-3.1-8B` requires a Hugging Face access token with Meta licence approval. 
---

## Pipeline

Run stages in order. Each stage depends on the previous.

### Stage 1 — Setup
```bash
qsub initial_set_up.sh
qsub hf_models/model_download.sh
qsub download_harmbench_evaluator_model.sh
qsub download_llama3.8_8b.sh
```

### Stage 2 — Raw Prompts
```bash
qsub download_harmbench_raw_prompts.sh
```
Downloads the HarmBench dataset (235 harmful behaviours) to `data/pre_refined_prompt/`.

### Stage 3 — Prompt Refinement *(wait for Stage 2)*
```bash
qsub run_refinement_all.sh
```
Produces two refined prompt files in `data/refined_prompt_data/`:
- `harmbench_refined.json` - LLM-refined prompts for DIJA
- `harmbench_context_nesting_refined.json` - deterministically template-wrapped prompts for Context Nesting

### Stage 4 — Generation *(wait for Stage 3)*
```bash
qsub harmbench_vanilla_gen.sh          # Vanilla baseline (no attack)
qsub harmbench_dija_gen.sh             # DIJA attack
qsub harmbench_context_nesting_gen.sh  # Context Nesting attack
qsub harmless_context_nesting_gen.sh   # Benign control (Context Nesting)
```
Outputs are saved to `results/<attack>/<defence>/generation.json`.

### Stage 5 — Evaluation *(wait for each gen job)*
```bash
qsub harmbench_vanilla_eval.sh
qsub harmbench_dija_eval.sh
qsub harmbench_context_nesting_eval.sh
qsub harmless_context_nesting_eval.sh
```
Each eval script runs four steps in sequence: ASR_e → ASR_k → PPL → SAR.

### Stage 6 — Aggregate
```bash
python collect_summary.py
```
Aggregates all results into `results/summary.json`.

---

## Metrics

| Metric | Description | Tool |
|---|---|---|
| **ASR_e** | Attack Success Rate (classifier-based) | HarmBench-Llama-2-13b-cls |
| **ASR_k** | Attack Success Rate (keyword-based) | 50 refusal phrase list |
| **PPL** | Perplexity of generated response | Llama-3.1-8B |
| **SAR** | Structure Adherence Rate (template following) | Rule-based, no model needed |

---

## Defences Evaluated

Each attack is evaluated against the following defence configurations:

- **No Defence** - default greedy low-confidence remasking
- **Fully Random** - uniform random token sampling at all mask positions
- **Stochastic Annealing** - random blend decayed over steps (α₀=0.3)
- **Stochastic Annealing (Exp.)** - exponential decay variant (α₀=0.9, c=0.12)
- **SPD** — Sequential Prefix Demasking (k=5)
- **SPD + Self-Reminder** - SPD with a safety system prompt
- **+ Block Audit** - any of the above combined with hidden-state self-detection (λ=0.1) and self-correction (8 steps, γ=0.9)

---
