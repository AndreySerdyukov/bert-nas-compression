"""The IMDB corpus and the exact split the notebooks used.

Everything that reads the dataset goes through here, so there is one place where the split is
defined and one place where it is checked. The check matters more than it looks: the split is
positional - `train_test_split` shuffles indices 0 to 49 999 - so it depends on the row order of the
CSV as much as on the seed. A different copy of "the IMDB dataset" silently produces a different
test set while every number downstream still looks plausible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _REPO_ROOT / "data" / "IMDB Dataset.csv"

# The Kaggle file the notebooks read, pinned by content. Not a security measure - it is how we know
# the split lands on the same 15 000 reviews the original evaluated on.
EXPECTED_CSV_SHA256 = "dfc447764f82be365fa9c2beef4e8df89d3919e3da95f5088004797d79695aa2"

# Reproduced from the notebooks, which all use these two calls with this seed.
SEED = 42
EXPECTED_SIZES = (28_000, 7_000, 15_000)

# The notebooks map the string labels this way round, and every checkpoint was trained with it.
# Confirmed empirically against a loaded model: class 1 is positive.
LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


class CorpusError(RuntimeError):
    """The corpus is missing, or is not the file the split was defined against."""


@dataclass(frozen=True)
class Split:
    """The three partitions, plus the fingerprint that proves which rows are in the test set."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    test_index_sha256: str

    @property
    def sizes(self) -> tuple[int, int, int]:
        return (len(self.train), len(self.val), len(self.test))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(path: Path = DEFAULT_CSV, *, verify: bool = True) -> pd.DataFrame:
    """Read the 50 000-row CSV, with its label column mapped to 0/1."""
    if not path.exists():
        raise CorpusError(f"{path} not found. See data/README.md, or run scripts/fetch_imdb.py.")
    if verify:
        actual = file_sha256(path)
        if actual != EXPECTED_CSV_SHA256:
            raise CorpusError(
                f"{path.name} is not the file this project's split was defined against.\n"
                f"  expected sha256 {EXPECTED_CSV_SHA256}\n"
                f"  got             {actual}\n"
                "The split is positional, so a different row order gives a different test set "
                "while every downstream number still looks plausible. Pass verify=False only if "
                "you know what that costs."
            )

    frame = pd.read_csv(path)
    if list(frame.columns) != ["review", "sentiment"]:
        raise CorpusError(f"expected columns ['review', 'sentiment'], got {list(frame.columns)}")
    frame = frame.assign(label=frame["sentiment"].map(LABEL_TO_ID))
    if frame["label"].isna().any():
        raise CorpusError("the sentiment column holds values other than positive/negative")
    return frame


def make_split(frame: pd.DataFrame) -> Split:
    """The notebooks' split, reproduced exactly and then checked.

    Two calls, in this order, with this seed. Both the sizes and a digest of the sorted test index
    are asserted: a pandas or scikit-learn release that changed the shuffling would otherwise
    quietly move the goalposts.
    """
    train_val, test = train_test_split(frame, test_size=0.3, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.2, random_state=SEED)

    sizes = (len(train), len(val), len(test))
    if sizes != EXPECTED_SIZES:
        raise CorpusError(f"split produced {sizes}, expected {EXPECTED_SIZES}")

    fingerprint = hashlib.sha256(
        ",".join(str(index) for index in sorted(test.index)).encode("ascii")
    ).hexdigest()

    return Split(train=train, val=val, test=test, test_index_sha256=fingerprint)


def load_split(path: Path = DEFAULT_CSV, *, verify: bool = True) -> Split:
    """Convenience: read the corpus and split it."""
    return make_split(load_corpus(path, verify=verify))
