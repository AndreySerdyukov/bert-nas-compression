# BERT NAS Compression

**What Neural Architecture Search actually buys when you compress a BERT classifier - measured,
with the controls that make the measurement mean something.** Four NAS strategies were used to cut a
109.5M-parameter sentiment model down to 4-5 encoder layers. This repository serves those models,
re-measures them honestly, and explains the method in ten chapters you can click through.

`FastAPI` · `React + TypeScript` · `PyTorch` · `transformers` · layer-mask NAS over BERT-base

> **Rebuild in progress.** The original academic project is preserved unchanged in
> [`notebooks/`](notebooks/); this is the application being built around it. The benchmark table
> below is the *original's* result, reproduced here as the thing being checked - not as this
> project's answer. See [Status](#status).

> Sibling projects: [ML Playground](https://github.com/AndreySerdyukov/ml-playground) - six tabular
> models behind one metadata-driven UI, and
> [DL Playground](https://github.com/AndreySerdyukov/dl-playground) - the same idea for PyTorch
> across image, text and tabular input.

---

## What this is

A team of three spent a term compressing `bert-base-uncased` for binary sentiment on IMDB, using
four Neural Architecture Search strategies implemented from scratch. For three of them the search
space is a single 12-bit mask: **which of the twelve encoder layers to keep**. The parameter count
follows from the mask alone -

```
params(k) = 24 429 314 + 7 087 872 · k
```

- so the whole cost side of the trade is arithmetic, and only the quality side needs a GPU. That
asymmetry is what the application is built around.

Two halves, both first-class:

- **A playground.** Paste a review, run it through every model at once, and watch a 4-layer network
  agree with the full one until it does not. A button finds the reviews where they disagree, because
  "2.5 points of accuracy" means more when it is fifty specific reviews you can read.
- **A methodology section.** Ten chapters on what NAS is, what each of the four methods does, and
  what happened when they were run - with the interactive pieces inline. Toggle layers and watch the
  cost change; move the fitness weights and watch the winner move; step a Bayesian surrogate through
  its own acquisition loop.

## Why it needed rebuilding

The original produced four mutually contradictory result tables and measured latency without a
warm-up. More importantly, it compared the compressed models against **three home-made baselines**
that disagree with each other by 0.8 points - so a 2.5-point drop could not be separated from
run-to-run variance.

This rebuild keeps one baseline and adds the controls the question actually needs:

| Control | The question it answers |
|---|---|
| Evenly-spaced / first-k / last-k masks | Did *searching* beat "keep every third layer"? |
| Random masks of the same size, several seeds | Is the found mask in the tail of the distribution, or the middle? |
| DistilBERT | How does distillation compare to search at a similar size? |
| TF-IDF + logistic regression | How much of this task needs a transformer at all? |

Every control is fine-tuned under the same protocol as the shipped NAS winners - one epoch on the
same 28 000 training rows at the same sequence length - and measured on the same test split by the
same harness. Nobody's reported figure is quoted, [and there is a specific reason for
that](data/README.md#-this-split-overlaps-the-official-imdb-partition).

It is entirely possible that a naive mask matches the searched one. If so, that is the result, and
it gets published as the result.

## Measured here

Generated from `backend/data/benchmark.json` by
[`scripts/render_readme_table.py`](backend/scripts/render_readme_table.py), which CI re-runs with
`--check`. A number in this table cannot change without being re-measured first.

<!-- benchmark:start -->

| Model | Method | Accuracy | Macro F1 | Params | Latency, median | Throughput | Memory |
|---|---|---:|---:|---:|---:|---:|---:|
| BERT-base | fine-tune (baseline) | 0.9330 | 0.9330 | 109 483 778 | 20.7 ms | 128/s | 386 MB |
| Random Search | random-search | 0.9077 | 0.9077 | 59 868 674 | 8.9 ms | 299/s | 183 MB |
| AlphaNAS | alphanas | 0.9027 | 0.9026 | 52 780 802 | 7.2 ms | 362/s | 147 MB |
| BANANAS | bananas | 0.9031 | 0.9031 | 52 780 802 | 7.2 ms | 369/s | 152 MB |
| AdaBERT | adabert | 0.9028 | 0.9028 | 7 814 146 | 0.9 ms | 1444/s | 88 MB |

Measured on the full 15,000 row test split (index sha256 `33d7fee10704`) by [`training/benchmark.py`](backend/training/benchmark.py).

**Accuracy** on `mps`. **Latency** is single-example, batch 1, on `cpu` with 1 thread: 5 warm-up passes discarded, median of 25 repeats, models interleaved round-robin. **Throughput** is a batch of 16 and is a different quantity - dividing it by the batch size does not give the latency column. Darwin arm64, torch 2.13.0, transformers 5.14.1.

All five timed the same review, but not on the same amount of work: BERT-base 33, Random Search 33, AlphaNAS 33, BANANAS 33, AdaBERT 128 tokens. The models that mask their padding see the review itself; AdaBERT has no attention mask and is served at a fixed 128, so its column is a shorter time over more tokens rather than a shorter time over the same ones.

**Memory** is the resident set of a process holding one loaded and warmed model, less the 411 MB that torch and transformers occupy before any model is loaded - a floor every model pays and none of them owns. Each is weighed in its own process: weighed one after another in one process, a freed model's pages go back to Python's allocator rather than to the OS and the next model looks nearly free. The figures run below the fp32 weight size because a safetensors checkpoint is mapped rather than copied, and pages nothing reads never become resident.

<!-- benchmark:end -->

## What the original reported

The head-to-head from `notebooks/results/NAS_Results.ipynb` - the only one of its four tables that
scores every model on one test set with one harness. **These are the numbers being checked, not
numbers this project stands behind.** They are kept here so the comparison above has something to
be a comparison with.

| Model | Method | Accuracy | Params | ms/example* |
|---|---|---:|---:|---:|
| BERT-base | fine-tune (baseline) | 0.9330 | 109 483 778 | 0.72 |
| Random Search | NAS, 5 layers | 0.9077 | 59 868 674 | 0.35 |
| BANANAS | Bayesian NAS, 4 layers | 0.9031 | 52 780 802 | 0.31 |
| AlphaNAS | evolutionary NAS, 4 layers | 0.9027 | 52 780 802 | 0.30 |
| AdaBERT | differentiable NAS | 0.8801 | 7 814 146 | 0.02 |

\* Not single-example latency. The figure is mean wall-clock per batch divided by 16, taken with no
warm-up and no CUDA synchronisation - a throughput number reported as a latency one. The replacement
measures both, names them differently, and publishes median and p95.

Two of the rows are worth reading twice. **AdaBERT's search degenerated to parameter-free
operations**, so the 7.8M model is a bag of embeddings rather than a distilled transformer - the
checkpoint size on the Hub confirms it to the byte. And **Random Search reports one architecture and
ships another**: the write-up says `[0,1,6,8,10]`, the uploaded model is `[0,1,5,7,9]`. Both facts,
and eight more, are in [`notebooks/README.md`](notebooks/README.md#what-did-not-reproduce). They are
findings about methods and scripts, not about people.

## Models

Fine-tuned checkpoints live on the Hugging Face Hub and are downloaded on demand - they are not in
this repository and never were.

| Name | Layers | Params | Source |
|---|---:|---:|---|
| `bert-imdb` | 12 | 109 483 778 | [AndreySerdyukov/bert-imdb](https://huggingface.co/AndreySerdyukov/bert-imdb) |
| `random-search` | 5 | 59 868 674 | [AndreySerdyukov/random-search](https://huggingface.co/AndreySerdyukov/random-search) |
| `alphanas` | 4 | 52 780 802 | [AndreySerdyukov/alphanas](https://huggingface.co/AndreySerdyukov/alphanas) |
| `bananas` | 4 | 52 780 802 | [alinaselivanets/bananas-bert](https://huggingface.co/alinaselivanets/bananas-bert) |
| `adabert` | n/a | 7 814 146 | [ilkonz/dnas](https://huggingface.co/ilkonz/dnas) |

**None of these carry `id2label`.** Class polarity is therefore established empirically at download
time, pinned into the manifest, re-checked on every load, and asserted in the test suite. Serving
inverted sentiment is the worst available failure mode, because from the outside it looks perfectly
healthy.

## Run locally

```bash
# Backend
cd backend
uv venv --python 3.12
uv pip install -e . --group dev
python scripts/fetch_models.py         # ~1.1 GB from the Hub, once
uv run uvicorn app.main:create_app --factory --reload   # http://127.0.0.1:8000

# Frontend (another terminal)
cd frontend
npm install && npm run dev             # http://127.0.0.1:5173, /api proxied to :8000
```

**A clean clone works without the download.** The methodology section, the architecture explorer's
cost arithmetic and the whole test suite need no weights at all; the endpoints that need a
prediction answer `503` and say which script to run. If port 8000 is taken,
`BACKEND_URL=http://localhost:8010 npm run dev` points the proxy elsewhere.

## Run in Docker

```bash
docker compose up --build              # frontend :3000, backend :8000
```

The image ships without weights - they are gitignored, so the build is identical whether or not the
machine building it has downloaded anything. Mount them in to get predictions:

```bash
docker compose run --rm -v "$PWD/backend/models:/app/models:ro" -p 8000:8000 backend
```

## Development

```bash
cd backend  && ruff check . && ruff format --check . && mypy app training && pytest
cd frontend && npm run lint && npm run format:check && npm run build && npm run check:deps && npm run test:e2e
```

`npm run check:deps` enforces two claims: the runtime dependency list is exactly three packages, and
no MDX runtime reached the bundle. The methodology chapters are `.mdx` compiled at build time, which
keeps ten chapters of prose out of JSX without shipping a markdown parser to every visitor - but
one configuration line would silently undo that, so it is checked rather than trusted.

## Repository layout

| Path | What's in it |
|---|---|
| `backend/app/` | The service: `api/` → `services/` → `repositories/` + `serving/` |
| `backend/training/` | The benchmark harness, the reference fine-tunes, the notebook trajectory extractor |
| `backend/data/` | Committed: the evaluation sample, the architectures, the search trajectories, the measured results |
| `frontend/src/content/` | The ten methodology chapters and their interactive widgets |
| `notebooks/` | The original Colab notebooks, unchanged. See [`notebooks/README.md`](notebooks/README.md) |
| `papers/` | The four reference papers and the final presentation |
| `data/` | Where the IMDB corpus goes. See [`data/README.md`](data/README.md) |

## Status

**The methodology section is complete and needs no weights at all** - ten chapters, three
interactive widgets, and every figure drawn from data parsed out of the notebooks and verified
against them in CI. Along with it: the data spine (`architectures.json`, `search_trajectories.json`,
`reported_results.json`), the mask arithmetic in both languages, 49 backend tests and 18 browser
tests.

Next: model serving and the playground, then the honest benchmark with its controls, then the
architecture explorer.

## Credits

*Numerical Optimization Methods*, spring 2025 - Andrey Serdyukov, Alina Selivanets, Ilya Konzafarov.
Contributions are broken down in [`notebooks/README.md`](notebooks/README.md#the-project); the
rebuild is mine. Dataset: IMDB 50k, Maas et al. 2011.
