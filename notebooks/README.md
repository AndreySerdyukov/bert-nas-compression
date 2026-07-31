# The original notebooks

These eleven notebooks are the academic project this repository was rebuilt from, **kept exactly as
they were written**. Nothing here has been re-run, re-ordered, translated or tidied. They are the
primary source: every number the application publishes can be traced back to a cell in this
directory, and every disagreement between them is listed below rather than quietly resolved.

The production path is `backend/` - the notebooks are not imported by the app and are not executed
by CI. What CI *does* do is parse them: `backend/training/extract_trajectories.py --check` re-reads
the printed search logs and fails if the committed `search_trajectories.json` no longer matches, so
the charts in the methodology section cannot drift away from their source.

## The project

*Numerical Optimization Methods*, spring 2025. Compressing a fine-tuned BERT-base sentiment
classifier with Neural Architecture Search, four strategies implemented from scratch.

| | Andrey Serdyukov | Alina Selivanets | Ilya Konzafarov |
|---|:-:|:-:|:-:|
| BERT fine-tuning | yes | yes | yes |
| Random Search | yes | | |
| AlphaNAS | yes | | |
| BANANAS | | yes | |
| Differentiable NAS (AdaBERT) | | | yes |

**Task.** Binary sentiment on IMDB 50k. **Base model.** `bert-base-uncased`, 12 encoder layers,
109 483 778 parameters. **Split.** `train_test_split(df, test_size=0.3, random_state=42)` then
`train_test_split(train, test_size=0.2, random_state=42)` - 28 000 train / 7 000 validation /
15 000 test. Every notebook uses the same two calls with the same seed, which is what makes the
head-to-head in `results/` meaningful.

**Search space.** For three of the four methods it is one-dimensional: a 12-bit mask over the
encoder layers, keeping between 4 and 12 of them. Attention heads, hidden size (768) and FFN size
(3072) were considered and deliberately left out. Parameter count follows from the mask alone:

```
params(k) = 24 429 314 + 7 087 872 · k
```

AdaBERT is the exception - it searches over operations, not layers.

## Directory guide

| Path | Contents |
|---|---|
| `bert_finetune/` | Three independent full-BERT fine-tunes, one per author |
| `random_search/` | Random Search: `_Selection` searches, `_Eval` trains and scores the winner. [Li & Talwalkar 2019](https://arxiv.org/abs/1902.07638) |
| `alphanas/` | Evolutionary search over the mask. [αNAS 2022](https://arxiv.org/abs/2205.03960) |
| `bananas/` | Surrogate-model Bayesian optimization. [White et al. 2019](https://arxiv.org/abs/1910.11858) |
| `dnas/` | Differentiable NAS with distillation. [AdaBERT](https://arxiv.org/abs/2001.04246) |
| `results/` | `NAS_Results.ipynb` - the head-to-head, seven checkpoints on one test set |

Each method directory keeps its tech report alongside its notebooks. The four reference papers and
the final presentation are in [`papers/`](../papers/).

## What the winners were

| Method | Layer mask | Layers kept | Params |
|---|---|---|---:|
| Random Search | `[1,1,0,0,0,1,0,1,0,1,0,0]` | `[0, 1, 5, 7, 9]` | 59 868 674 |
| AlphaNAS | `[0,1,1,0,0,0,1,0,0,0,0,1]` | `[1, 2, 6, 11]` | 52 780 802 |
| BANANAS | `[1,1,0,0,0,0,1,0,0,1,0,0]` | `[0, 1, 6, 9]` | 52 780 802 |
| AdaBERT | five cells, all `avg_pool` | n/a | 7 814 146 |

## What did not reproduce

This section exists because the rebuild had to pick numbers to publish, and picking them meant
reading every source. Each item is a property of a method or a script, not of a person.

**1. Four mutually contradictory result tables.** The same seven models are reported with different
accuracies in the eval notebooks, in `results/NAS_Results.ipynb`, in the four tech reports and in
the presentation. Only `NAS_Results.ipynb` scores every model on one common test set with one
harness, so that is the table the rebuild treats as the original's canonical result - and it is the
one the new measurements are compared against.

**2. Random Search reports one architecture and ships another.** `RandomSearch_Selection.ipynb`
concludes with `[0, 1, 6, 8, 10]`, and the tech report repeats it. `RandomSearch_Eval.ipynb` - the
notebook that trained and uploaded the published checkpoint - sets
`layers_flag = [1,1,0,0,0,1,0,1,0,1,0,0]`, which is `[0, 1, 5, 7, 9]`. Both keep five layers, so the
parameter count in every table is unaffected; the architecture named in the write-up is not the one
on the Hub. The application therefore reports `[0, 1, 5, 7, 9]`, and a unit test pins the
distinction so it cannot be silently "corrected" later.

**3. The AlphaNAS final re-evaluation trains on its own evaluation set.** The last call passes
`val_loader` as both the training and the evaluation loader. That run reports 0.7679 - below the
0.75 floor the search itself would have rejected - against 0.8536 for the same mask during the
search. The published checkpoint comes from `AlphaNas_Eval.ipynb`, which does not have this
problem, so the shipped model is unaffected; the number printed at the end of the selection
notebook is.

**4. BANANAS searched on 1 000 rows at `max_length=128`.** Every other method used 5 000-50 000 rows
at 512. Candidate accuracies during that search land between 0.58 and 0.69, and evaluating the
*identical* configuration twice produced 0.6214 and 0.5786 - a 4.3 pp spread wider than the entire
range the search was trying to rank. The final print also reports the last iteration's accuracy
rather than the best configuration's. The winning mask was then trained properly in
`BANANAS_Eval.ipynb`, and that model is sound; what is not supported is the claim that the search
found it *because* it searched well.

**5. Search budgets differ by a factor of fifty, so the methods are not comparable to each other.**
Random Search evaluated 3 candidates on 5 000 rows; AlphaNAS 12 on 10 000; BANANAS 10 on 1 000;
AdaBERT ran 15 epochs on 49 582. Any ranking of "which NAS method is better" from these runs is
reading noise. The rebuild does not make that claim, and adds naive and random layer masks trained
under a fixed protocol so that the more answerable question - did searching beat a simple rule? -
can actually be asked.

**6. The AdaBERT search degenerated to a parameter-free architecture.** Gumbel-softmax selected
`avg_pool` for all five surviving cells. `avg_pool` has no parameters, so the resulting network is
`Embedding(30522, 256) → mean over tokens → Linear(256, 2)`: a bag-of-embeddings classifier, not a
distilled transformer. The 7 814 146 parameter count is exactly `30522·256 + 256·2 + 2`, and the
31 258 949-byte checkpoint on the Hub confirms it. The notebook attributes the collapse to running
15 search epochs where the literature uses 50-90, which is a fair reading - the efficiency term
dominates the loss at ~12.1 throughout and barely moves. The model is a legitimate and genuinely
fast baseline; it is just not the method its name implies.

**7. `FrozenAdaBERT` is defined twice, differently.** The class in `dnas/` stacks the five
`avg_pool` cells; the one redefined in `results/` drops them and goes straight from embedding to
mean-pool. `avg_pool` is parameter-free, so `load_state_dict` succeeds either way and neither
version errors - but the forward passes differ, which is the most likely explanation for the same
checkpoint scoring 0.8979 in one notebook and 0.8801 in the other. The application loads with
`strict=True` so the artifact itself decides which definition is right, rather than the loader
guessing.

**8. The latency figures are not measurements of what they are labelled.** `avg_sample_time_ms` is
mean wall-clock per batch divided by the batch size, with no warm-up and no CUDA synchronisation.
That is a throughput-derived figure being reported as single-example latency, and on a GPU without
synchronisation the timer can stop before the work does. The 0.02 ms attributed to AdaBERT is below
what the harness can resolve. The rebuild measures both quantities, names them differently, and
publishes median and p95 rather than a mean of one pass.

**9. Nothing runs outside Colab.** Every notebook mounts Google Drive and reads
`/content/drive/MyDrive/IMDB Dataset.csv`. There is no requirements file, no package structure and
no seeding beyond the split. Reproducing any of it means having the author's Drive.

**10. The split overlaps the official IMDB partition.** The Kaggle CSV is the standard 50 000
reviews - 25 000 official train plus 25 000 official test - and the notebooks reshuffle all of it
with one seed. The resulting 15 000-row test set therefore contains rows from the official training
half, and the models were trained on rows from the official test half. These checkpoints cannot be
compared against published IMDB numbers. It is why the rebuild fine-tunes and measures every
external reference itself, on this same split, instead of quoting anyone's reported figure.

## Attribution

The dataset is [IMDB 50k](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
(Maas et al., 2011). See [`data/README.md`](../data/README.md).
