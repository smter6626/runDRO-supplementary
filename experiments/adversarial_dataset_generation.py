"""Generate adversarially perturbed DEBD test splits via greedy bit-flip search."""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fastcircuits import Circuit

DEBD_DIR = Path("DEBD/datasets")
LIKELIHOOD_PC_DIR = Path("likelihood_learned_pcs")
SAMPLE_DRO_PC_DIR = Path("circuit_sample_dro_learned_pcs")
DROSPN_PC_DIR = Path("drospn_learned_pcs")
OUT_DIR = Path("adversarially_perturbed_datasets")
PC_SOURCES = ("likelihood_learned_pcs", "circuit_sample_dro_learned_pcs", "drospn_learned_pcs")
BLOCK_SIZE = 4
SEED = 0
PROGRESS_EVERY = 100
CHUNK_DIVISOR = 8
DEFAULT_WORKERS = min(os.cpu_count() or 1, 8)

_WORKER_PC: Circuit | None = None


def load_split(dataset: str, split: str) -> np.ndarray:
    path = DEBD_DIR / dataset / f"{dataset}.{split}.data"
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.int8)


def save_dataset(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row.tolist())


def debd_datasets() -> list[str]:
    return sorted(
        d.name
        for d in DEBD_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.test.data").is_file()
    )


def out_prefix(pc_source: str) -> str:
    if pc_source == "likelihood_learned_pcs":
        return "likelihood_learned_pcs"
    if pc_source == "circuit_sample_dro_learned_pcs":
        return "circuit_sample_dro"
    return "drospn"


def pc_path(dataset: str, pc_source: str, k: int) -> Path:
    if pc_source == "likelihood_learned_pcs":
        base = LIKELIHOOD_PC_DIR
    elif pc_source == "circuit_sample_dro_learned_pcs":
        base = SAMPLE_DRO_PC_DIR
    else:
        base = DROSPN_PC_DIR
    path = base / "hclt" / dataset / str(BLOCK_SIZE)
    if pc_source in ("circuit_sample_dro_learned_pcs", "drospn_learned_pcs"):
        path /= f"K{k}"
    return path / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"


def out_path(dataset: str, k: int, pc_source: str) -> Path:
    return (
        OUT_DIR
        / out_prefix(pc_source)
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}_dataset_K{k}.data"
    )


def datapoint_to_assignment(x: np.ndarray) -> dict[int, int]:
    return {i: int(x[i]) for i in range(len(x))}


def greedy_corrupt(pc: Circuit, x: np.ndarray, k: int) -> np.ndarray:
    x = x.copy()
    assignment = {i: int(x[i]) for i in range(len(x))}
    current_ll = pc.log_likelihood(assignment)
    for _ in range(k):
        best_ll, best_i = current_ll, 0
        for i in range(len(x)):
            assignment[i] ^= 1
            ll = pc.log_likelihood(assignment)
            if ll < best_ll:
                best_ll, best_i = ll, i
            assignment[i] ^= 1
        assignment[best_i] ^= 1
        x[best_i] ^= 1
        current_ll = best_ll
    return x


def _init_worker(pc_path_str: str) -> None:
    global _WORKER_PC
    _WORKER_PC = Circuit.load(pc_path_str)


def _corrupt_row(args: tuple[np.ndarray, int]) -> np.ndarray:
    row, k = args
    return greedy_corrupt(_WORKER_PC, row, k)


def _log_row0_ll(
    pc: Circuit, dataset: str, k: int, original: np.ndarray, corrupted: np.ndarray
) -> None:
    ll_before = pc.log_likelihood(datapoint_to_assignment(original))
    ll_after = pc.log_likelihood(datapoint_to_assignment(corrupted))
    print(f"[{dataset}] row 0 LL {ll_before:.4f} -> {ll_after:.4f} (k={k})")


def generate_dataset(
    dataset: str, k: int, workers: int, pc_source: str
) -> None:
    output = out_path(dataset, k, pc_source)
    if output.exists():
        print(f"Skipping {dataset} ({output} exists)")
        return

    pc_file = pc_path(dataset, pc_source, k)
    if not pc_file.exists():
        print(f"Skipping {dataset} (missing PC: {pc_file})")
        return

    test_data = load_split(dataset, "test")
    path_str = str(pc_file)

    if workers == 1:
        pc = Circuit.load(path_str)
        corrupted = np.empty_like(test_data)
        for i, row in enumerate(test_data):
            corrupted[i] = greedy_corrupt(pc, row, k)
            if i == 0:
                _log_row0_ll(pc, dataset, k, row, corrupted[i])
            if (i + 1) % PROGRESS_EVERY == 0:
                print(f"[{dataset}] {i + 1}/{len(test_data)} rows")
    else:
        chunksize = max(1, len(test_data) // (workers * CHUNK_DIVISOR))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(path_str,),
        ) as executor:
            corrupted = np.stack(
                list(
                    executor.map(
                        _corrupt_row,
                        ((row, k) for row in test_data),
                        chunksize=chunksize,
                    )
                )
            )
        pc = Circuit.load(path_str)
        _log_row0_ll(pc, dataset, k, test_data[0], corrupted[0])
        print(f"[{dataset}] {len(test_data)} rows ({workers} workers)")

    save_dataset(output, corrupted)
    print(f"Saved {output} ({len(corrupted)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate adversarially perturbed DEBD test datasets."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of greedy bit-flip steps per datapoint.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel worker processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="DEBD dataset names (default: all with test splits).",
    )
    parser.add_argument(
        "--pc-source",
        choices=PC_SOURCES,
        default="likelihood_learned_pcs",
        help=(
            "PC directory to adversarially optimize against "
            "(default: likelihood_learned_pcs)."
        ),
    )
    args = parser.parse_args()

    if args.k < 0:
        parser.error("--k must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    datasets = args.datasets if args.datasets else debd_datasets()
    print(
        f"Generating adversarial datasets for {len(datasets)} dataset(s), "
        f"pc_source={args.pc_source}, k={args.k}, workers={args.workers}"
    )
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        generate_dataset(dataset, args.k, args.workers, args.pc_source)


if __name__ == "__main__":
    main()
