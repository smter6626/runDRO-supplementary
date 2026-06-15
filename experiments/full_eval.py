"""Compare mean log-likelihood on adversarially-perturbed test splits."""

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
BLOCK_SIZE = 4
SEED = 0


def likelihood_pc_path(dataset: str) -> Path:
    return (
        LIKELIHOOD_PC_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


def sample_dro_pc_paths(dataset: str, k: int) -> list[Path]:
    base = SAMPLE_DRO_PC_DIR / "hclt" / dataset / str(BLOCK_SIZE) / f"K{k}"
    stem = f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}"
    return [
        base / f"{stem}_best_ll_adv.json",
        base / f"{stem}.json",
    ]


def resolve_sample_dro_pc_path(dataset: str, k: int) -> Path | None:
    for path in sample_dro_pc_paths(dataset, k):
        if path.exists():
            return path
    return None


def drospn_pc_path(dataset: str, k: int) -> Path:
    return (
        DROSPN_PC_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


def test_path(dataset: str) -> Path:
    return DEBD_DIR / dataset / f"{dataset}.test.data"


def adv_path(dataset: str, k: int, adv_prefix: str) -> Path:
    return (
        ADV_DIR
        / adv_prefix
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}_dataset_K{k}.data"
    )


def load_dataset(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.int8)


def datapoint_to_assignment(x: np.ndarray) -> dict[int, int]:
    return {i: int(x[i]) for i in range(len(x))}


def mean_log_likelihood(pc: Circuit, data: np.ndarray) -> float:
    total = 0.0
    for row in data:
        total += pc.log_likelihood(datapoint_to_assignment(row))
    return total / len(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare likelihood-learned, sample-DRO, and DROSPN PCs, each on "
            "its own adversarially-perturbed test split."
        )
    )
    parser.add_argument("dataset", help="DEBD dataset name (e.g. nltcs).")
    parser.add_argument("k", type=int, help="Adversarial perturbation budget K.")
    args = parser.parse_args()

    if args.k < 0:
        parser.error("k must be non-negative")

    likelihood_path = likelihood_pc_path(args.dataset)
    sample_dro_path = resolve_sample_dro_pc_path(args.dataset, args.k)
    drospn_path = drospn_pc_path(args.dataset, args.k)
    test_data_path = test_path(args.dataset)
    likelihood_data_path = adv_path(args.dataset, args.k, "likelihood_learned_pcs")
    sample_dro_data_path = adv_path(args.dataset, args.k, "circuit_sample_dro")
    drospn_data_path = adv_path(args.dataset, args.k, "drospn")

    if sample_dro_path is None:
        parser.error(
            "missing sample-DRO PC: "
            f"tried {', '.join(str(p) for p in sample_dro_pc_paths(args.dataset, args.k))}"
        )

    for label, path in (
        ("likelihood PC", likelihood_path),
        ("sample-DRO PC", sample_dro_path),
        ("DROSPN PC", drospn_path),
        ("test dataset", test_data_path),
        ("likelihood adversarial dataset", likelihood_data_path),
        ("sample-DRO adversarial dataset", sample_dro_data_path),
        ("DROSPN adversarial dataset", drospn_data_path),
    ):
        if not path.exists():
            parser.error(f"missing {label}: {path}")

    test_data = load_dataset(test_data_path)
    likelihood_data = load_dataset(likelihood_data_path)
    sample_dro_data = load_dataset(sample_dro_data_path)
    drospn_data = load_dataset(drospn_data_path)
    likelihood_pc = Circuit.load(likelihood_path)
    sample_dro_pc = Circuit.load(sample_dro_path)
    drospn_pc = Circuit.load(drospn_path)

    ll_likelihood_test = mean_log_likelihood(likelihood_pc, test_data)
    ll_sample_dro_test = mean_log_likelihood(sample_dro_pc, test_data)
    ll_drospn_test = mean_log_likelihood(drospn_pc, test_data)
    ll_likelihood_adv = mean_log_likelihood(likelihood_pc, likelihood_data)
    ll_sample_dro_adv = mean_log_likelihood(sample_dro_pc, sample_dro_data)
    ll_drospn_adv = mean_log_likelihood(drospn_pc, drospn_data)
    ll_sample_dro_on_likelihood_adv = mean_log_likelihood(sample_dro_pc, likelihood_data)
    ll_drospn_on_likelihood_adv = mean_log_likelihood(drospn_pc, likelihood_data)

    print(f"dataset: {args.dataset}")
    print(f"adversarial K: {args.k}")
    print(f"rows: {len(test_data)}")
    print(f"mean LL (likelihood PC on test data): {ll_likelihood_test:.6f}")
    print(f"mean LL (sample-DRO PC on test data): {ll_sample_dro_test:.6f}")
    print(f"mean LL (DROSPN PC on test data): {ll_drospn_test:.6f}")
    print(f"delta (sample-DRO - likelihood, test): {ll_sample_dro_test - ll_likelihood_test:+.6f}")
    print(f"delta (DROSPN - likelihood, test): {ll_drospn_test - ll_likelihood_test:+.6f}")
    print(f"delta (DROSPN - sample-DRO, test): {ll_drospn_test - ll_sample_dro_test:+.6f}")
    print(
        f"mean LL (sample-DRO PC on sample-DRO adv data): "
        f"{ll_sample_dro_adv:.6f}"
    )
    print(f"mean LL (DROSPN PC on DROSPN adv data): {ll_drospn_adv:.6f}")
    print(f"delta (sample-DRO - likelihood, adv): {ll_sample_dro_adv - ll_likelihood_adv:+.6f}")
    print(f"delta (DROSPN - likelihood, adv): {ll_drospn_adv - ll_likelihood_adv:+.6f}")
    print(f"delta (DROSPN - sample-DRO, adv): {ll_drospn_adv - ll_sample_dro_adv:+.6f}")
    print(f"mean LL (likelihood PC on likelihood adv data): {ll_likelihood_adv:.6f}")
    print(f"mean LL (sample-DRO PC on likelihood adv data): {ll_sample_dro_on_likelihood_adv:.6f}")
    print(f"mean LL (DROSPN PC on likelihood adv data): {ll_drospn_on_likelihood_adv:.6f}")
    print(
        f"delta (sample-DRO - likelihood, likelihood adv): "
        f"{ll_sample_dro_on_likelihood_adv - ll_likelihood_adv:+.6f}"
    )
    print(
        f"delta (DROSPN - likelihood, likelihood adv): "
        f"{ll_drospn_on_likelihood_adv - ll_likelihood_adv:+.6f}"
    )
    print(
        f"delta (DROSPN - sample-DRO, likelihood adv): "
        f"{ll_drospn_on_likelihood_adv - ll_sample_dro_on_likelihood_adv:+.6f}"
    )


if __name__ == "__main__":
    main()
