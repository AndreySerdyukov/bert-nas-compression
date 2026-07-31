# Data

## The corpus

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) -
50 000 movie reviews labelled `positive` / `negative`, two columns: `review`, `sentiment`.
Originally from Maas et al., *Learning Word Vectors for Sentiment Analysis*, ACL 2011.

The CSV is **not committed** - it is 66 MB and it is not ours to redistribute. Put it at
`data/IMDB Dataset.csv`, or let the fetch script do it:

```bash
cd backend
python scripts/fetch_imdb.py              # writes ../data/IMDB Dataset.csv
```

`load_dataset("imdb")` is **not** a drop-in substitute, despite what the original `data/README.md`
suggested: the Hugging Face dataset has different column names (`text` / `label`) and ships the
official 25k/25k partition, so it reproduces neither the columns the notebooks read nor the split
they made.

Only two things need the raw corpus: `training/benchmark.py --full` (scoring all 15 000 test rows)
and `training/train_reference.py` (fine-tuning the reference models). Everything else - the app, the
test suite, CI, the Docker image - runs off the committed 2 000-row evaluation sample described
below. A clean clone is fully usable without downloading anything.

## The split

Reproduced exactly from the notebooks, and asserted rather than assumed:

```python
train_val, test = train_test_split(df, test_size=0.3, random_state=42)
train, val      = train_test_split(train_val, test_size=0.2, random_state=42)
# 28 000 / 7 000 / 15 000
```

`training/benchmark.py` checks the three sizes **and a sha256 of the sorted test-set index** before
it measures anything. A pandas or scikit-learn release that changed the shuffling would otherwise
silently score a different 15 000 rows while every number still looked plausible.

## ⚠️ This split overlaps the official IMDB partition

The Kaggle file is the standard corpus: 25 000 official train rows plus 25 000 official test rows,
concatenated. The notebooks reshuffle all 50 000 with one seed, so the 15 000-row test set contains
rows from the official *training* half - and, symmetrically, the models were fine-tuned on rows from
the official *test* half.

Two consequences, both load-bearing for how this repository reports numbers:

1. **Accuracies here cannot be compared with published IMDB results.** Anything quoting the official
   25k/25k split is measuring something else. Our figures are internally consistent and externally
   incomparable.
2. **Every external reference model is fine-tuned and measured by us, on this same split.** We never
   quote a reported figure for DistilBERT or anything else - a reported figure comes from the clean
   split and would flatter or penalise it for the wrong reason.

Fixing the split would mean retraining all five checkpoints, which is GPU-hours we do not have, and
would also make the rebuild stop describing the project that actually happened. Publishing the
limitation is the honest alternative.

## What is committed

| Path | What it is |
|---|---|
| `backend/data/imdb_eval_sample.jsonl` | 2 000 rows drawn from the 15 000-row test split, stratified, seeded, source indices checksummed. Built by `training/build_eval_sample.py`. The app, CI and the Docker image all score these identical rows |
| `backend/data/label_probe.json` | A small set of unambiguous reviews used to establish which class index means "positive". None of the checkpoints carry `id2label`, so polarity is measured, pinned in the manifest, and re-checked at every model load |

The sample is committed rather than sampled at runtime for one reason: a latency or accuracy figure
means nothing unless the rows behind it are the same everywhere.
