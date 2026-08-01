"""Put the IMDB corpus where this project's split expects it, and prove it is the right file.

The CSV is 66 MB and is not ours to redistribute, so it is not committed. Only two things need it -
`training/benchmark.py` scoring the full split and `training/train_reference.py` training the
controls - and everything else in the repository runs off the committed 2 000-row sample.

**The file is pinned by content, and that is the point of this script.** The split is positional:
`train_test_split` with a fixed seed over a fixed row order. A corpus with the same 50 000 reviews in
a different order produces a different 15 000-row test set while every downstream number still looks
entirely plausible, so "some IMDB CSV" is not a substitute for this one. The sha256 lives in
`training/dataset.py` and is checked here before the file is accepted.

**`load_dataset("imdb")` is not a drop-in replacement either**, despite what the original project's
README suggested: the Hugging Face copy has different column names (`text` / `label`) and ships the
official 25k/25k partition rather than the concatenation Kaggle distributes. It reproduces neither
the columns the notebooks read nor the split they made, so this script does not offer it as a
fallback - a fallback that silently changes the test set is worse than no fallback.

Kaggle needs credentials, which this script does not invent: it reads `KAGGLE_USERNAME` and
`KAGGLE_KEY`, or `~/.kaggle/kaggle.json`, and when it finds neither it prints where to click and
where to put the file. Nothing here writes credentials anywhere.

Run (from the backend directory):
    python scripts/fetch_imdb.py            # download, verify, install
    python scripts/fetch_imdb.py --check    # verify what is already there, download nothing
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

from training.dataset import DEFAULT_CSV, EXPECTED_CSV_SHA256, file_sha256

DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
MEMBER = "IMDB Dataset.csv"

WHERE_TO_GET_IT = f"""
No Kaggle credentials found, and the corpus cannot be redistributed from here.

Either set credentials and re-run this script:

    export KAGGLE_USERNAME=... KAGGLE_KEY=...        # Kaggle -> Settings -> Create New Token
    # or place that token at ~/.kaggle/kaggle.json

or download it by hand and drop the file in place:

    https://www.kaggle.com/datasets/{DATASET}
    unzip it and put "{MEMBER}" at {DEFAULT_CSV}

Then `python scripts/fetch_imdb.py --check` confirms it is the right file.
""".strip()


def credentials() -> tuple[str, str] | None:
    """Kaggle username and key, from the environment or the token file. Never written back."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key

    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not token.exists():
        return None
    try:
        payload = json.loads(token.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  ! {token} is not valid JSON", file=sys.stderr)
        return None
    user, key = payload.get("username"), payload.get("key")
    return (user, key) if user and key else None


def verify(path: Path) -> bool:
    """Is the file at `path` the corpus the split was defined against?"""
    if not path.exists():
        print(f"not there: {path}")
        return False
    actual = file_sha256(path)
    size_mb = path.stat().st_size / 1e6
    if actual == EXPECTED_CSV_SHA256:
        print(f"ok: {path} ({size_mb:.0f} MB), sha256 matches")
        return True
    print(f"WRONG FILE: {path} ({size_mb:.0f} MB)", file=sys.stderr)
    print(f"  expected sha256 {EXPECTED_CSV_SHA256}", file=sys.stderr)
    print(f"  got             {actual}", file=sys.stderr)
    print(
        "  The split is positional, so a different row order silently changes which 15 000 "
        "reviews every model is scored on.",
        file=sys.stderr,
    )
    return False


def download(user: str, key: str, into: Path) -> Path:
    """Fetch the dataset archive and return the extracted CSV inside `into`."""
    token = base64.b64encode(f"{user}:{key}".encode()).decode("ascii")
    request = urllib.request.Request(DOWNLOAD_URL, headers={"Authorization": f"Basic {token}"})

    archive = into / "imdb.zip"
    print(f"downloading {DATASET} …")
    with urllib.request.urlopen(request) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"  {archive.stat().st_size / 1e6:.0f} MB")

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if MEMBER not in names:
            raise SystemExit(f"the archive holds {names}, expected {MEMBER!r}")
        bundle.extract(MEMBER, path=into)
    return into / MEMBER


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify what is already there and exit"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if args.check:
        return 0 if verify(args.out) else 1

    if args.out.exists():
        if verify(args.out):
            print("nothing to do")
            return 0
        # Not overwritten: it may be a corpus someone is using deliberately, and this script is
        # not the right place to decide that.
        print(f"\nmove or delete {args.out} first", file=sys.stderr)
        return 1

    found = credentials()
    if found is None:
        print(WHERE_TO_GET_IT, file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as workspace:
        staging = Path(workspace)
        try:
            extracted = download(*found, staging)
        except urllib.error.HTTPError as error:
            hint = " (bad credentials?)" if error.code in (401, 403) else ""
            print(f"Kaggle answered {error.code}{hint}", file=sys.stderr)
            return 1
        except urllib.error.URLError as error:
            print(f"could not reach Kaggle: {error.reason}", file=sys.stderr)
            return 1

        # Verified in staging, so a wrong file never lands at the path everything else reads.
        if not verify(extracted):
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), args.out)

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
