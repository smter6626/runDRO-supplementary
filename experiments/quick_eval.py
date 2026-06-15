"""Print mean log-likelihood on test, adversarial, and random perturbation data."""

import argparse
import csv
from pathlib import Path

import numpy as np

from fastcircuits import Circuit

DEBD_DIR = Path("DEBD/datasets")
LIKELIHOOD_PC_DIR = Path("likelihood_learned_pcs")
SAMPLE_DRO_PC_DIR = Path("circuit_sample_dro_learned_pcs")
DROSPN_PC_DIR = Path("drospn_learned_pcs")
ADV_DIR = Path("adversarially_perturbed_datasets")
RAND_DIR = Path("randomly_perturbed_datasets")
BLOCK_SIZE = 4
SEED = 0


def load_dataset(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.int8)


def mean_log_likelihood(pc: Circuit, data: np.ndarray) -> float:
    total = 0.0
    for row in data:
        total += pc.log_likelihood({i: int(row[i]) for i in range(len(row))})
    return total / len(data)


def try_load(path: Path) -> Circuit | None:
    if not path.exists():
        return None
    return Circuit.load(path)


def sample_dro_pc_paths(dataset: str, k: int) -> list[Path]:
    base = SAMPLE_DRO_PC_DIR / "hclt" / dataset / str(BLOCK_SIZE) / f"K{k}"
    stem = f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}"
    return [
        base / f"{stem}_best_ll_adv.json",
        base / f"{stem}.json",
    ]


def try_load_sample_dro_pc(dataset: str, k: int) -> Circuit | None:
    for path in sample_dro_pc_paths(dataset, k):
        pc = try_load(path)
        if pc is not None:
            return pc
    return None


def print_mean_ll(label: str, pc: Circuit | None, data: np.ndarray) -> None:
    if pc is None:
        print(f"mean LL ({label}): SKIPPED")
        return
    print(f"mean LL ({label}): {mean_log_likelihood(pc, data):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="DEBD dataset name (e.g. nltcs)")
    parser.add_argument("k", type=int, help="Adversarial perturbation budget K")
    args = parser.parse_args()

    dataset = args.dataset
    k = args.k

    likelihood_pc = try_load(
        LIKELIHOOD_PC_DIR / "hclt" / dataset / str(BLOCK_SIZE)
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )
    sample_dro_pc = try_load_sample_dro_pc(dataset, k)
    drospn_pc = try_load(
        DROSPN_PC_DIR / "hclt" / dataset / str(BLOCK_SIZE) / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )

    test_data = load_dataset(DEBD_DIR / dataset / f"{dataset}.test.data")
    adv_data = load_dataset(
        ADV_DIR / "likelihood_learned_pcs" / "hclt" / dataset / str(BLOCK_SIZE) / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}_dataset_K{k}.data"
    )
    rand_data = load_dataset(RAND_DIR / f"K{k}" / dataset / f"{dataset}.test.data")

    print_mean_ll("likelihood PC, test", likelihood_pc, test_data)
    print_mean_ll("sample-DRO PC, test", sample_dro_pc, test_data)
    print_mean_ll("DROSPN PC, test", drospn_pc, test_data)
    print_mean_ll("likelihood PC, likelihood adv", likelihood_pc, adv_data)
    print_mean_ll("sample-DRO PC, likelihood adv", sample_dro_pc, adv_data)
    print_mean_ll("DROSPN PC, likelihood adv", drospn_pc, adv_data)
    print_mean_ll("likelihood PC, random", likelihood_pc, rand_data)
    print_mean_ll("sample-DRO PC, random", sample_dro_pc, rand_data)
    print_mean_ll("DROSPN PC, random", drospn_pc, rand_data)


if __name__ == "__main__":
    main()
