# BERT NAS Compression

**What Neural Architecture Search actually bought when four strategies compressed a BERT sentiment
classifier - measured, against the controls that make the measurement mean something.**

`FastAPI` · `React + TypeScript` · `PyTorch` · `transformers` · layer-mask NAS over BERT-base

Sibling projects: [ML Playground](https://github.com/AndreySerdyukov/ml-playground) ·
[DL Playground](https://github.com/AndreySerdyukov/dl-playground)

---

## The finding

A team of three spent a term cutting `bert-base-uncased` down to four or five encoder layers with
NAS, and compared the results against each other. This rebuild compared them against controls
instead - eighteen of them, trained under the same protocol on the same split.

- **TF-IDF with logistic regression scores higher than all four searched models** (0.9141 against
  0.9027-0.9077), in ten seconds of CPU and with no transformer at all.
- **DistilBERT gives up 0.35 points to full BERT where the best searched model gives up 2.53**, at
  the same parameter count to within 1 536 parameters.
- **Keeping the bottom k layers beats the search at both depths.** It is written `range(k)`.
- Against five random masks per depth, the four-layer results sit **below the median**.

A Wilson interval on 15 000 rows is about ±0.46 points, so the careful version is not that naive
rules win. It is that **the result of the search is indistinguishable from a random mask of the same
depth** - and the search cost days of compute where the random mask costs one line.

None of this says the searches were built wrong. They were implemented from scratch and they work.
It says the comparison that would have told anyone what they were worth was never run.

## What you can do with it

- **Playground** - paste a review, run every model on it at once, and find the reviews where a
  four-layer network stops agreeing with the full one.
- **Explorer** - cut layers out of the fine-tuned model and watch accuracy collapse to 50.00% on
  2 000 reviews. Beside it, the same mask trained properly: 90.62%. That gap is what training each
  candidate buys, and why a search cannot skip it.
- **Benchmark** - every number this project publishes, the eighteen controls beside them, and a
  verdict per encoder depth.
- **Methodology** - ten chapters on what NAS is, what each method did, and what the audit of the
  original notebooks found. Works with no weights downloaded at all.

The search space is a 12-bit mask over encoder layers, so the cost side of the trade is arithmetic:

```
params(k) = 24 429 314 + 7 087 872 · k
```

## Measured here

Generated from `backend/data/benchmark.json` and `backend/data/controls.json` by
[`scripts/render_readme_table.py`](backend/scripts/render_readme_table.py), which CI re-runs with
`--check`. A number here cannot change without being re-measured first.

<!-- benchmark:start -->

| Model | Method | Accuracy | Macro F1 | Params | Latency, median | Throughput | Memory |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT-base | fine-tune (baseline) | 0.9330 | 0.9330 | 109 483 778 | 20.7 ms | 128/s | 386 MB |
| Random Search | random-search | 0.9077 | 0.9077 | 59 868 674 | 8.9 ms | 299/s | 183 MB |
| AlphaNAS | alphanas | 0.9027 | 0.9026 | 52 780 802 | 7.2 ms | 362/s | 147 MB |
| BANANAS | bananas | 0.9031 | 0.9031 | 52 780 802 | 7.2 ms | 369/s | 152 MB |
| AdaBERT | adabert | 0.9028 | 0.9028 | 7 814 146 | 0.9 ms | 1444/s | 88 MB |

Measured on the full 15 000 row test split (index sha256 `33d7fee10704`) by [`training/benchmark.py`](backend/training/benchmark.py).

**Accuracy** on `mps`. **Latency** is single-example on `cpu`, 1 thread, 5 warm-ups discarded, median of 25, round-robin. **Throughput** is a batch of 16 and a different quantity: dividing it by the batch size does not give the latency column. Darwin arm64, torch 2.13.0, transformers 5.14.1.

All five timed the same review over different amounts of work: BERT-base 33, Random Search 33, AlphaNAS 33, BANANAS 33, AdaBERT 128 tokens. AdaBERT has no attention mask and is served at a fixed 128, so its column is a shorter time over *more* tokens.

**Memory** is one model per process, less the 411 MB floor torch and transformers occupy before any model loads. Weighed one after another in a single process instead, a freed model's pages return to Python's allocator rather than to the OS and every model after the first looks nearly free.

### The controls

| Control | Layers | Accuracy | Macro F1 |
|---|---|---:|---:|
| DistilBERT | - | 0.9295 | 0.9295 |
| TF-IDF + logistic regression | - | 0.9141 | 0.9141 |
| First 5 layers | 0,1,2,3,4 | 0.9114 | 0.9114 |
| Random 5 layers, seed 2 | 0,1,2,5,10 | 0.9099 | 0.9099 |
| Random 5 layers, seed 4 | 1,3,4,6,7 | 0.9087 | 0.9087 |
| Evenly spaced, 5 layers | 0,3,6,8,11 | 0.9080 | 0.9079 |
| Random 5 layers, seed 3 | 2,3,5,8,9 | 0.9067 | 0.9067 |
| Random 4 layers, seed 1 | 1,2,4,9 | 0.9065 | 0.9065 |
| First 4 layers | 0,1,2,3 | 0.9062 | 0.9061 |
| Random 5 layers, seed 1 | 1,2,4,9,10 | 0.9061 | 0.9061 |
| Random 4 layers, seed 4 | 1,3,4,6 | 0.9057 | 0.9057 |
| Random 4 layers, seed 2 | 0,1,5,10 | 0.9039 | 0.9038 |
| Random 5 layers, seed 0 | 0,4,6,7,11 | 0.9023 | 0.9023 |
| Random 4 layers, seed 3 | 2,3,8,9 | 0.9003 | 0.9002 |
| Evenly spaced, 4 layers | 0,4,7,11 | 0.8959 | 0.8958 |
| Random 4 layers, seed 0 | 0,4,6,11 | 0.8940 | 0.8939 |
| Last 5 layers | 7,8,9,10,11 | 0.8851 | 0.8850 |
| Last 4 layers | 8,9,10,11 | 0.8799 | 0.8799 |

They answer 5 questions:

1. How much of this task needs a transformer at all?
2. Did searching beat spreading the layers evenly?
3. Does it matter which end of the stack the layers come from?
4. How does distillation compare with search at a similar size?
5. Is the found mask in the tail of the distribution, or the middle?

Trained under the shipped checkpoints' own protocol, read out of the eval notebooks: 1 epoch over all 28 000 training rows, 512 tokens, batch 16, AdamW at 3e-05 with weight decay 0.01 and a 10% warm-up. Nothing was tuned - shortening the training would bias every comparison on this page in this project's favour.

<!-- benchmark:end -->

## What the original reported

The head-to-head from `notebooks/results/NAS_Results.ipynb` - the only one of its four tables that
scores every model on one test set. **These are the numbers being checked, not numbers this project
stands behind.**

| Model | Method | Accuracy | Params | ms/example* |
|---|---|---:|---:|---:|
| BERT-base | fine-tune (baseline) | 0.9330 | 109 483 778 | 0.72 |
| Random Search | NAS, 5 layers | 0.9077 | 59 868 674 | 0.35 |
| BANANAS | Bayesian NAS, 4 layers | 0.9031 | 52 780 802 | 0.31 |
| AlphaNAS | evolutionary NAS, 4 layers | 0.9027 | 52 780 802 | 0.30 |
| AdaBERT | differentiable NAS | 0.8801 | 7 814 146 | 0.02 |

\* A throughput number reported as a latency: mean wall-clock per batch over 16, no warm-up, no
synchronisation. The replacement measures both, names them differently, and publishes median and p95.

Two rows repay a second look: **AdaBERT's search degenerated to parameter-free operations** (the
7.8M model is a bag of embeddings, not a distilled transformer), and **Random Search reports
`[0,1,6,8,10]` but ships `[0,1,5,7,9]`**. Both, and eight more, are in
[`notebooks/README.md`](notebooks/README.md#what-did-not-reproduce) - findings about methods and
scripts, never about people.

## Models

Checkpoints live on the Hugging Face Hub and are downloaded on demand.

| Name | Layers | Params | Source |
|---|---:|---:|---|
| `bert-imdb` | 12 | 109 483 778 | [AndreySerdyukov/bert-imdb](https://huggingface.co/AndreySerdyukov/bert-imdb) |
| `random-search` | 5 | 59 868 674 | [AndreySerdyukov/random-search](https://huggingface.co/AndreySerdyukov/random-search) |
| `alphanas` | 4 | 52 780 802 | [AndreySerdyukov/alphanas](https://huggingface.co/AndreySerdyukov/alphanas) |
| `bananas` | 4 | 52 780 802 | [alinaselivanets/bananas-bert](https://huggingface.co/alinaselivanets/bananas-bert) |
| `adabert` | n/a | 7 814 146 | [ilkonz/dnas](https://huggingface.co/ilkonz/dnas) |

**None carry `id2label`**, so class polarity is measured at download time, pinned into the manifest
and re-checked on every load. Serving inverted sentiment is the worst failure available here,
because from outside it looks perfectly healthy.

## Run it

```bash
# Backend
cd backend
uv venv --python 3.12 && uv pip install -e . --group dev
python scripts/fetch_models.py                           # ~1.1 GB from the Hub, once
uv run uvicorn app.main:create_app --factory --reload     # :8000

# Frontend, another terminal
cd frontend && npm install && npm run dev                  # :5173, /api proxied to :8000
```

**A clean clone works without the download**: the methodology half and the whole test suite need no
weights, and endpoints that need a prediction answer `503` naming the script to run. Port taken?
`BACKEND_URL=http://localhost:8010 npm run dev`.

`docker compose up --build` serves the frontend on :3000. The image copies in whatever
`backend/models/` holds when it is built - nothing on a clean clone, a gigabyte after the fetch
script has run. To mount the checkpoints instead of baking them in, uncomment the `volumes:` block
in [`docker-compose.yml`](docker-compose.yml). The IMDB corpus is needed only to re-run the
benchmark or retrain the controls - `python scripts/fetch_imdb.py`, see
[`data/README.md`](data/README.md).

## Development

```bash
cd backend  && ruff check . && ruff format --check . && mypy app training scripts && pytest
cd frontend && npm run lint && npm run format:check && npm run build && npm run check:deps && npm run test:e2e
```

158 backend tests, 46 browser tests, three CI jobs. `check:deps` enforces two claims that one
configuration line would silently undo: exactly three runtime dependencies, and no MDX runtime in
the bundle.

## Repository layout

| Path | What's in it |
|---|---|
| `backend/app/` | The service: `api/` → `services/` → `repositories/` + `serving/` |
| `backend/training/` | Benchmark harness, the eighteen controls, the notebook extractor |
| `backend/data/` | Committed measurements and the evaluation sample |
| `frontend/src/content/` | The ten methodology chapters and their widgets |
| `notebooks/` | The original Colab notebooks, unchanged. See [`notebooks/README.md`](notebooks/README.md) |
| `papers/` | The four reference papers and the final presentation |

## Credits

*Numerical Optimization Methods*, spring 2025 - Andrey Serdyukov, Alina Selivanets, Ilya Konzafarov.
Contributions are broken down in [`notebooks/README.md`](notebooks/README.md#the-project); the
rebuild is mine. Dataset: IMDB 50k, Maas et al. 2011.
