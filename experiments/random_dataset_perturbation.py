"""Randomly flip K bits in each row of every DEBD test split."""

import argparse
import csv
from pathlib import Path

import numpy as np

DEBD_DIR = Path("DEBD/datasets")
OUT_DIR = Path("randomly_perturbed_datasets")
SEED = 0


def load_test_split(dataset: str) -> np.ndarray:
    path = DEBD_DIR / dataset / f"{dataset}.test.data"
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.int8)


def save_dataset(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row.tolist())


def debd_test_datasets() -> list[str]:
    return sorted(
        d.name
        for d in DEBD_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.test.data").is_file()
    )


def flip_k_bits(row: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    out = row.copy()
    n = len(out)
    for _ in range(k):
        out[rng.integers(n)] ^= 1
    return out


def perturb_dataset(dataset: str, k: int, seed: int) -> None:
    test_data = load_test_split(dataset)
    rng = np.random.default_rng(seed)
    perturbed = np.stack([flip_k_bits(row, k, rng) for row in test_data])

    output = OUT_DIR / f"K{k}" / dataset / f"{dataset}.test.data"
    save_dataset(output, perturbed)
    print(f"Saved {output} ({len(perturbed)} rows, {test_data.shape[1]} bits)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly perturb DEBD test datasets by flipping K bits per row."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of random bit flips per datapoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"RNG seed (default: {SEED}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="DEBD dataset names (default: all with test splits).",
    )
    args = parser.parse_args()

    if args.k < 0:
        parser.error("--k must be non-negative")

    datasets = args.datasets if args.datasets else debd_test_datasets()
    print(f"Perturbing {len(datasets)} dataset(s) with k={args.k}, seed={args.seed}")
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        perturb_dataset(dataset, args.k, args.seed)


if __name__ == "__main__":
    main()
