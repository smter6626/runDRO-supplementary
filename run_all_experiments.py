"""Run the reviewer-facing runDRO supplementary experiment pipeline."""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence, Union

PAPER_DATASETS = [
    "nltcs",
    "msnbc",
    "plants",
    "bnetflix",
    "dna",
    "tmovie",
    "bbc",
]
H_VALUES = [1, 3, 5]
RESULT_MODELS = ["likelihood", "sample_dro", "drospn"]
RESULT_SPLITS = ["test", "adv", "random"]
DEFAULT_ETA_THETA = 1e-3
DEFAULT_ETA_PHI = 1e-3
DEFAULT_THETA_NUM_SAMPLES = 2000
DEFAULT_WORKERS = min(os.cpu_count() or 1, 8)
RANDOM_SEED = 0
SMOKE_DATASETS = ["nltcs"]
SMOKE_H_VALUES = [1]
SMOKE_OURS_ITERS = 1
SMOKE_THETA_NUM_SAMPLES = 16
SMOKE_WORKERS = 1
SMOKE_EPOCHS = 1

PACKAGE_ROOT = Path(__file__).resolve().parent
DEBD_ROOT = Path("DEBD/datasets")
MLE_OUTPUT_ROOT = Path("learned_pcs")
LIKELIHOOD_OUTPUT_ROOT = Path("likelihood_learned_pcs")
OUTPUT_ROOTS = [
    MLE_OUTPUT_ROOT,
    LIKELIHOOD_OUTPUT_ROOT,
    Path("adversarially_perturbed_datasets"),
    Path("circuit_sample_dro_learned_pcs"),
    Path("drospn_learned_pcs"),
    Path("randomly_perturbed_datasets"),
]


class RunnerError(RuntimeError):
    """Raised for fail-clearly pipeline precondition and artifact errors."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a reduced composition check: nltcs, h=1, one training epoch "
            "for MLE/DRO-SPN, one Ours iteration, reduced theta samples, one worker."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="DEBD dataset identifiers to run (full-run default: seven paper datasets).",
    )
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=int,
        default=None,
        help="Wasserstein h values to run (full-run default: 1 3 5).",
    )
    parser.add_argument(
        "--ours-iters",
        type=int,
        default=None,
        help=(
            "Outer theta iterations for Ours/sample-DRO; required unless --smoke "
            f"is used (smoke default: {SMOKE_OURS_ITERS})."
        ),
    )
    parser.add_argument(
        "--eta-theta",
        type=float,
        default=DEFAULT_ETA_THETA,
        help=f"Theta learning rate for Ours (default: {DEFAULT_ETA_THETA:g}).",
    )
    parser.add_argument(
        "--eta-phi",
        type=float,
        default=DEFAULT_ETA_PHI,
        help=f"Phi learning rate for Ours (default: {DEFAULT_ETA_PHI:g}).",
    )
    parser.add_argument(
        "--theta-num-samples",
        type=int,
        default=None,
        help=(
            "Samples from Q_phi per theta update for Ours "
            f"(full-run default: {DEFAULT_THETA_NUM_SAMPLES}; "
            f"smoke default: {SMOKE_THETA_NUM_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Workers for likelihood-PC adversarial splits "
            f"(full-run default: {DEFAULT_WORKERS}; smoke default: {SMOKE_WORKERS})."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results.csv"),
        help="CSV output path (default: outputs/results.csv).",
    )
    parser.add_argument(
        "--output-latex",
        type=Path,
        default=None,
        help="Optionally write the existing LaTeX table to this path.",
    )
    args = parser.parse_args()
    apply_run_mode_defaults(args, parser)

    if args.ours_iters < 1:
        parser.error("--ours-iters must be at least 1")
    if any(h < 1 for h in args.h_values):
        parser.error("--h-values must all be at least 1")
    if args.eta_theta <= 0:
        parser.error("--eta-theta must be positive")
    if args.eta_phi <= 0:
        parser.error("--eta-phi must be positive")
    if args.theta_num_samples < 1:
        parser.error("--theta-num-samples must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.output_latex is not None and args.output_latex == args.output_csv:
        parser.error("--output-latex must differ from --output-csv")
    return args


def apply_run_mode_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.smoke:
        apply_smoke_overrides(args)
        return

    if args.datasets is None:
        args.datasets = PAPER_DATASETS
    if args.h_values is None:
        args.h_values = H_VALUES
    if args.ours_iters is None:
        parser.error("--ours-iters is required unless --smoke is used")
    if args.theta_num_samples is None:
        args.theta_num_samples = DEFAULT_THETA_NUM_SAMPLES
    if args.workers is None:
        args.workers = DEFAULT_WORKERS


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if args.datasets is None:
        args.datasets = SMOKE_DATASETS
    if args.h_values is None:
        args.h_values = SMOKE_H_VALUES
    if args.ours_iters is None:
        args.ours_iters = SMOKE_OURS_ITERS
    if args.theta_num_samples is None:
        args.theta_num_samples = SMOKE_THETA_NUM_SAMPLES
    if args.workers is None:
        args.workers = SMOKE_WORKERS


def apply_smoke_epoch_override(args: argparse.Namespace, module) -> None:
    if not args.smoke:
        return
    module.EPOCHS = SMOKE_EPOCHS


def configure_sample_dro(args: argparse.Namespace, sample_dro) -> None:
    sample_dro.ETA_THETA = args.eta_theta
    sample_dro.ETA_PHI = args.eta_phi
    sample_dro.THETA_NUM_SAMPLES = args.theta_num_samples


def ensure_unique(values: Union[Sequence[str], Sequence[int]], label: str) -> None:
    if len(values) != len(set(values)):
        raise RunnerError(f"{label} must not contain duplicates: {values}")


def ensure_absent_or_empty(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise RunnerError(f"expected output root to be a directory: {path}")
    if any(path.iterdir()):
        raise RunnerError(
            f"output root already exists and is not empty: {path}. "
            "Move it aside before running from scratch."
        )


def ensure_output_path_available(path: Path) -> None:
    if path.exists():
        raise RunnerError(f"refusing to overwrite existing output file: {path}")
    if path.parent != Path(".") and path.parent.exists() and not path.parent.is_dir():
        raise RunnerError(f"output parent is not a directory: {path.parent}")


def verify_debd_inputs(datasets: Sequence[str]) -> None:
    missing: list[Path] = []
    for dataset in datasets:
        for split in ("train", "valid", "test"):
            path = DEBD_ROOT / dataset / f"{dataset}.{split}.data"
            if not path.is_file():
                missing.append(path)
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise RunnerError(f"missing DEBD input files:\n{detail}")


def mle_pc_path(dataset: str) -> Path:
    return (
        MLE_OUTPUT_ROOT
        / "hclt"
        / dataset
        / "4"
        / f"hclt_{dataset}_blocksize4_seed0.json"
    )


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        detail = "\n".join(f"  - {path}" for path in missing)
        raise RunnerError(f"missing {label} artifacts:\n{detail}")


def require_sample_dro_outputs(datasets: Sequence[str], h: int, table_module) -> None:
    missing: list[str] = []
    for dataset in datasets:
        paths = table_module.sample_dro_pc_paths(dataset, h)
        if not any(path.is_file() for path in paths):
            missing.append(f"{dataset} h={h}: " + " or ".join(str(path) for path in paths))
    if missing:
        detail = "\n".join(f"  - {item}" for item in missing)
        raise RunnerError(f"missing Ours/sample-DRO artifacts:\n{detail}")


def handoff_mle_to_likelihood() -> None:
    if not MLE_OUTPUT_ROOT.is_dir() or not any(MLE_OUTPUT_ROOT.iterdir()):
        raise RunnerError(f"missing MLE output directory: {MLE_OUTPUT_ROOT}")
    if LIKELIHOOD_OUTPUT_ROOT.exists():
        if not LIKELIHOOD_OUTPUT_ROOT.is_dir():
            raise RunnerError(f"handoff target is not a directory: {LIKELIHOOD_OUTPUT_ROOT}")
        if any(LIKELIHOOD_OUTPUT_ROOT.iterdir()):
            raise RunnerError(
                f"handoff target already exists and is not empty: {LIKELIHOOD_OUTPUT_ROOT}"
            )

    # Pipeline path handoff: the MLE learner writes learned_pcs/, while the
    # downstream paper scripts read likelihood_learned_pcs/.
    shutil.copytree(MLE_OUTPUT_ROOT, LIKELIHOOD_OUTPUT_ROOT, dirs_exist_ok=True)


def missing_result_values(
    results: dict,
    datasets: Sequence[str],
    h_values: Sequence[int],
) -> list[str]:
    missing: list[str] = []
    for dataset in datasets:
        for h in h_values:
            for split in RESULT_SPLITS:
                for model in RESULT_MODELS:
                    if results[dataset][h][split][model] is None:
                        missing.append(f"{dataset}, h={h}, split={split}, model={model}")
    return missing


def verify_csv(path: Path, expected_rows: int) -> None:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    expected_header = ["dataset", "h", "method", "T", "T_a", "T_r"]
    if header != expected_header:
        raise RunnerError(f"unexpected CSV header in {path}: {header}")
    if len(rows) != expected_rows:
        raise RunnerError(
            f"unexpected CSV row count in {path}: got {len(rows)}, expected {expected_rows}"
        )
    incomplete = [
        row
        for row in rows
        if len(row) != len(expected_header) or any(v == "" for v in row)
    ]
    if incomplete:
        raise RunnerError(f"CSV contains incomplete rows in {path}")


def main() -> None:
    args = parse_args()
    datasets = list(args.datasets)
    h_values = list(args.h_values)
    ensure_unique(datasets, "datasets")
    ensure_unique(h_values, "h-values")

    os.chdir(PACKAGE_ROOT)
    verify_debd_inputs(datasets)
    for output_root in OUTPUT_ROOTS:
        ensure_absent_or_empty(output_root)
    ensure_output_path_available(args.output_csv)
    if args.output_latex is not None:
        ensure_output_path_available(args.output_latex)

    print(f"Running from {PACKAGE_ROOT}")
    print(f"Datasets: {' '.join(datasets)}")
    print(f"h values: {' '.join(str(h) for h in h_values)}")
    if args.smoke:
        print(
            "Smoke mode: "
            f"MLE EPOCHS={SMOKE_EPOCHS}, "
            f"DRO-SPN EPOCHS={SMOKE_EPOCHS}, "
            f"theta_num_samples={args.theta_num_samples}, workers={args.workers}"
        )

    import torch
    import learn_debd_pcs as mle_module

    apply_smoke_epoch_override(args, mle_module)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== MLE PC ===")
    for dataset in datasets:
        mle_module.learn_and_save(dataset, device)
    require_files((mle_pc_path(dataset) for dataset in datasets), "MLE PC")

    print("\n=== MLE path handoff ===")
    handoff_mle_to_likelihood()

    import experiments.generate_latex_table as table_module

    require_files(
        (table_module.likelihood_pc_path(dataset) for dataset in datasets),
        "likelihood PC",
    )

    for h in h_values:
        import experiments.adversarial_dataset_generation as adv_module

        print(f"\n=== h={h}: likelihood-PC adversarial splits ===")
        for dataset in datasets:
            adv_module.generate_dataset(dataset, h, args.workers, "likelihood_learned_pcs")
        require_files(
            (table_module.adv_path(dataset, h) for dataset in datasets),
            "adversarial split",
        )

        import experiments.circuit_sample_dro as sample_dro

        configure_sample_dro(args, sample_dro)
        print(f"\n=== h={h}: Ours/sample-DRO ===")
        for dataset in datasets:
            sample_dro.generate_circuit(dataset, args.ours_iters, h)
        require_sample_dro_outputs(datasets, h, table_module)

        print(f"\n=== h={h}: DRO-SPN ===")
        import learn_robust_debd_pcs as drospn_module

        apply_smoke_epoch_override(args, drospn_module)
        for dataset in datasets:
            drospn_module.learn_and_save(dataset, h, device)
        require_files(
            (table_module.drospn_pc_path(dataset, h) for dataset in datasets),
            "DRO-SPN",
        )

        import experiments.random_dataset_perturbation as random_module

        print(f"\n=== h={h}: random perturbation splits ===")
        for dataset in datasets:
            random_module.perturb_dataset(dataset, h, RANDOM_SEED)
        require_files(
            (table_module.random_path(dataset, h) for dataset in datasets),
            "random split",
        )

    print("\n=== Results ===")
    results = table_module.collect_results(datasets, h_values)
    missing = missing_result_values(results, datasets, h_values)
    if missing:
        detail = "\n".join(f"  - {item}" for item in missing)
        raise RunnerError(f"collect_results returned incomplete values:\n{detail}")

    table_module.write_csv(results, datasets, args.output_csv, h_values)
    expected_rows = len(datasets) * len(h_values) * len(RESULT_MODELS)
    verify_csv(args.output_csv, expected_rows)
    print(f"Wrote CSV results to {args.output_csv} ({expected_rows} rows)")

    if args.output_latex is not None:
        latex = table_module.render_latex(results, datasets, precision=2, h_values=h_values)
        args.output_latex.parent.mkdir(parents=True, exist_ok=True)
        args.output_latex.write_text(latex + "\n")
        print(f"Wrote LaTeX table to {args.output_latex}")


if __name__ == "__main__":
    try:
        main()
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
