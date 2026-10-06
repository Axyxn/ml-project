"""Download the ULB CSV from the mirror used in TensorFlow's official tutorial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from src.config import DATASET_MIRROR, DATASET_PAGE, ROOT
from src.data import DatasetError, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "creditcard.csv")
    args = parser.parse_args()
    if args.output.exists():
        df = load_dataset(args.output)
        print(f"Using existing validated CSV: {len(df):,} rows, {int(df.Class.sum()):,} frauds.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".csv.part")
    digest = hashlib.sha256()
    try:
        with requests.get(DATASET_MIRROR, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            count = 0
            print(f"Downloading real ULB transactions from {DATASET_MIRROR}", flush=True)
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if count // (20 * 1024 * 1024) != (count - len(chunk)) // (20 * 1024 * 1024):
                        print(f"  {count / 1024**2:.0f} MB downloaded (expected {total / 1024**2:.0f} MB)", flush=True)
        df = load_dataset(partial)
        if len(df) != 284_807 or int(df.Class.sum()) != 492:
            raise DatasetError("Mirror does not match the expected original 284,807 rows / 492 frauds.")
        partial.replace(args.output)
        metadata = {"source": DATASET_PAGE, "mirror": DATASET_MIRROR,
                    "sha256": digest.hexdigest(), "rows": len(df), "frauds": int(df.Class.sum())}
        args.output.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved validated dataset: {args.output}\nSHA256: {digest.hexdigest()}")
        return 0
    except (requests.RequestException, OSError, DatasetError) as exc:
        # A partial download is never mistaken for a valid training file.
        print(f"Download unavailable: {exc}\nManual fallback: visit {DATASET_PAGE}, download and unzip "
              f"creditcard.csv, then place it at {args.output}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
