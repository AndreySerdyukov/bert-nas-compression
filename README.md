# BERT Compression via Neural Architecture Search

Compressing a fine-tuned **BERT-base** text classifier with **Neural Architecture Search (NAS)** —
cutting parameters and inference cost while keeping accuracy within a **≤ 7 pp** budget.
Four NAS strategies are implemented from scratch and benchmarked head-to-head on the same task.

> Academic project — *Numerical Optimization Methods*, Spring 2025. Team of three
> (Andrey Serdyukov, Ilya Konzafarov, Alina Selivanets). **My contribution:** full-BERT
> fine-tuning baseline, **Random Search NAS**, and **AlphaNAS**. Trained checkpoints are
> published on the Hugging Face Hub (links below).

## Task & setup
- **Task:** binary sentiment classification on **IMDB** (28k train / 7k val / 15k test).
- **Baseline:** `bert-base-uncased`, 12 encoder layers, **109.5M** parameters.
- **Goal:** reduce parameter count and inference latency **without** dropping accuracy by more
  than 7 percentage points.
- **Search space:** which/how many encoder layers to keep (and, for AdaBERT, a distilled
  differentiable architecture).

## Results

Every model evaluated on the **same 15k test set**, single GPU, identical tokenizer & batching.

| Model | Method | Accuracy | Params | vs baseline | ms / example |
|---|---|---:|---:|---:|---:|
| **BERT-base** | fine-tune (baseline) | **93.3%** | 109.5M | — | 0.72 |
| Random Search | NAS (mine) | 90.8% | 59.9M | **−45%** params | 0.35 |
| AlphaNAS | NAS (mine) | 90.3% | 52.8M | **−52%** params | 0.30 |
| BANANAS | Bayesian NAS | 90.3% | 52.8M | **−52%** params | 0.31 |
| AdaBERT | Differentiable NAS | 88.0% | **7.8M** | **−93%** params | 0.02 |

*(Team fine-tuned three full-BERT baselines at 93.3–94.1%; the 93.3% run is used as the reference above.)*

### Key findings
- **Layer-selection NAS (Random Search / AlphaNAS / BANANAS)** halves the model (**~2× fewer
  parameters, ~2× faster**) for only a **2.5–3.0 pp** accuracy drop — comfortably inside the budget.
- **AdaBERT (differentiable NAS)** is the extreme point: a **14× smaller** model (7.8M params,
  **~34× faster** inference) that still holds **88%** accuracy — a −5.3 pp trade for a radical
  footprint reduction, ideal for edge / high-throughput serving.
- Bayesian search (BANANAS) and AlphaNAS converge to the same 4-layer architecture as the best
  random-search find here — evidence that on this space a well-tuned random search is a strong,
  cheap baseline (consistent with Li & Talwalkar, 2019).

## Trained models (Hugging Face Hub)
- [`AndreySerdyukov/bert-imdb`](https://huggingface.co/AndreySerdyukov/bert-imdb) — full BERT baseline
- [`AndreySerdyukov/random-search`](https://huggingface.co/AndreySerdyukov/random-search) — Random Search NAS (5 layers)
- [`AndreySerdyukov/alphanas`](https://huggingface.co/AndreySerdyukov/alphanas) — AlphaNAS (4 layers)

## Methods & references
| Dir | Method | Paper |
|---|---|---|
| `random_search/` | Random Search NAS | Li & Talwalkar, *Random Search and Reproducibility for NAS* (2019) |
| `bananas/` | BANANAS (Bayesian opt.) | White et al., *BANANAS* (2019) |
| `alphanas/` | AlphaNAS | *αNAS: Neural Architecture Search using Property-Guided Synthesis* (2022) |
| `dnas/` | Differentiable NAS | *AdaBERT: Task-Adaptive BERT Compression with Differentiable NAS* |

## Repository structure
```
bert_finetune/   # full BERT-base fine-tuning (per-author baselines)
random_search/   # Random Search NAS — selection + evaluation + tech report
alphanas/        # AlphaNAS — selection + evaluation + tech report
bananas/         # BANANAS — Bayesian NAS
dnas/            # AdaBERT / Differentiable NAS
results/         # NAS_Results.ipynb — the head-to-head benchmark above
paper/           # reference papers
performance/     # final presentation
```

## Reproduce
Each method folder has a `*_Selection.ipynb` (architecture search) and `*_Eval.ipynb`
(test-set evaluation). The end-to-end comparison lives in [`results/NAS_Results.ipynb`](results/NAS_Results.ipynb),
which loads every checkpoint from the Hub and reproduces the results table.
