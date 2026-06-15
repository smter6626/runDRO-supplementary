"""Generate a LaTeX results table of mean log-likelihood across PCs and splits.

For each dataset and K in {1, 3, 5}, evaluate three circuits (likelihood PC,
sample-DRO PC, DROSPN PC) on three test splits (clean test, likelihood
adversarial, random perturbation) and emit a LaTeX ``table`` block.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

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

K_VALUES = [1, 3, 5]
MODELS = ["likelihood", "sample_dro", "drospn"]
SPLITS = ["test", "adv", "random"]
CSV_METHOD_NAMES = {
    "likelihood": "MLE PC",
    "sample_dro": "Ours",
    "drospn": "DRO-SPN",
}

DEFAULT_DATASETS = ["plants", "bnetflix", "dna", "tmovie", "bbc"]

DISPLAY_NAMES = {
    "nltcs": "NLTCS",
    "msnbc": "MSNBC",
    "kdd": "KDDCup2k",
    "plants": "Plants",
    "baudio": "Audio",
    "jester": "Jester",
    "bnetflix": "Netflix",
    "accidents": "Accidents",
    "mushrooms": "Mushrooms",
    "adult": "Adult",
    "connect4": "Connect 4",
    "ocr_letters": "OCR Letters",
    "rcv1": "RCV-1",
    "tretail": "Retail",
    "pumsb_star": "Pumsb-star",
    "dna": "DNA",
    "kosarek": "Kosarek",
    "msweb": "MSWeb",
    "nips": "NIPS",
    "book": "Book",
    "tmovie": "Movie",
    "cwebkb": "WebKB",
    "cr52": "Reuters-52",
    "c20ng": "20 NewsGroup",
    "moviereview": "Movie reviews",
    "bbc": "BBC",
    "voting": "Voting",
    "ad": "Ad",
}


def _normalize_h_values(h_values: Optional[Sequence[int]]) -> list[int]:
    values = list(K_VALUES if h_values is None else h_values)
    if not values:
        raise ValueError("at least one h value is required")
    return values


def likelihood_pc_path(dataset: str) -> Path:
    return (
        LIKELIHOOD_PC_DIR / "hclt" / dataset / str(BLOCK_SIZE)
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


def sample_dro_pc_paths(dataset: str, k: int) -> list[Path]:
    base = SAMPLE_DRO_PC_DIR / "hclt" / dataset / str(BLOCK_SIZE) / f"K{k}"
    stem = f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}"
    return [
        base / f"{stem}_best_ll_adv.json",
        base / f"{stem}.json",
    ]


def try_load_sample_dro_pc(dataset: str, k: int) -> Circuit | None:
    for path in sample_dro_pc_paths(dataset, k):
        pc = try_load_circuit(path)
        if pc is not None:
            return pc
    return None


def drospn_pc_path(dataset: str, k: int) -> Path:
    return (
        DROSPN_PC_DIR / "hclt" / dataset / str(BLOCK_SIZE) / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


def test_path(dataset: str) -> Path:
    return DEBD_DIR / dataset / f"{dataset}.test.data"


def adv_path(dataset: str, k: int) -> Path:
    return (
        ADV_DIR / "likelihood_learned_pcs" / "hclt" / dataset / str(BLOCK_SIZE) / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}_dataset_K{k}.data"
    )


def random_path(dataset: str, k: int) -> Path:
    return RAND_DIR / f"K{k}" / dataset / f"{dataset}.test.data"


def load_dataset(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.int8)


def mean_log_likelihood(pc: Circuit, data: np.ndarray) -> float:
    total = 0.0
    for row in data:
        total += pc.log_likelihood({i: int(row[i]) for i in range(len(row))})
    return total / len(data)


def try_load_circuit(path: Path) -> Circuit | None:
    if not path.exists():
        return None
    return Circuit.load(path)


def try_load_data(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return load_dataset(path)


def collect_results(datasets: list[str], h_values: Optional[Sequence[int]] = None) -> dict:
    """Return results[dataset][k][split][model] = float | None."""
    h_values = _normalize_h_values(h_values)
    results: dict = {}
    for dataset in datasets:
        likelihood_pc = try_load_circuit(likelihood_pc_path(dataset))
        test_data = try_load_data(test_path(dataset))
        results[dataset] = {}
        for k in h_values:
            sample_dro_pc = try_load_sample_dro_pc(dataset, k)
            drospn_pc = try_load_circuit(drospn_pc_path(dataset, k))
            pcs = {
                "likelihood": likelihood_pc,
                "sample_dro": sample_dro_pc,
                "drospn": drospn_pc,
            }
            split_data = {
                "test": test_data,
                "adv": try_load_data(adv_path(dataset, k)),
                "random": try_load_data(random_path(dataset, k)),
            }
            results[dataset][k] = {}
            for split in SPLITS:
                data = split_data[split]
                results[dataset][k][split] = {
                    model: (
                        mean_log_likelihood(pc, data)
                        if pc is not None and data is not None
                        else None
                    )
                    for model, pc in pcs.items()
                }
    return results


def write_csv(
    results: dict,
    datasets: list[str],
    output_path: Path,
    h_values: Optional[Sequence[int]] = None,
) -> None:
    h_values = _normalize_h_values(h_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "h", "method", "T", "T_a", "T_r"])
        for dataset in datasets:
            for k in h_values:
                for model in MODELS:
                    writer.writerow(
                        [
                            dataset,
                            k,
                            CSV_METHOD_NAMES[model],
                            results[dataset][k]["test"][model],
                            results[dataset][k]["adv"][model],
                            results[dataset][k]["random"][model],
                        ]
                    )


def format_cell(
    value: float | None,
    bold: bool,
    underline: bool,
    precision: int,
) -> str:
    if value is None:
        return "---"
    text = f"{value:.{precision}f}"
    if bold:
        text = f"\\textbf{{{text}}}"
    if underline:
        text = f"\\underline{{{text}}}"
    return text


def rank_models(values: dict) -> tuple[set[str], set[str]]:
    """Return (best, second_best) model keys; ties at a tier are all included."""
    present = {m: v for m, v in values.items() if v is not None}
    if not present:
        return set(), set()
    tiers = sorted(set(present.values()), reverse=True)
    best_vals = {tiers[0]}
    second_vals = {tiers[1]} if len(tiers) >= 2 else set()
    best = {m for m, v in present.items() if v in best_vals}
    second = {m for m, v in present.items() if v in second_vals}
    return best, second


def render_split_cells(values: dict, precision: int) -> list[str]:
    best, second = rank_models(values)
    return [
        format_cell(
            values[model],
            model in best,
            model in second,
            precision,
        )
        for model in MODELS
    ]


def render_latex(
    results: dict,
    datasets: list[str],
    precision: int,
    h_values: Optional[Sequence[int]] = None,
) -> str:
    h_values = _normalize_h_values(h_values)
    lines: list[str] = []
    lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{multirow}")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{ll ccc ccc ccc}")
    lines.append("\\toprule")
    lines.append(
        " &  & \\multicolumn{3}{c}{$\\mathcal{T}$} & "
        "\\multicolumn{3}{c}{$\\mathcal{T}_a$} & "
        "\\multicolumn{3}{c}{$\\mathcal{T}_r$} \\\\"
    )
    lines.append("\\cmidrule(r){3-5} \\cmidrule(lr){6-8} \\cmidrule(l){9-11}")
    lines.append(
        "Dataset & $h$ & MLE PC & \\textbf{Ours} & DRO-SPN & MLE PC & \\textbf{Ours} & DRO-SPN & "
        "MLE PC & \\textbf{Ours} & DRO-SPN \\\\"
    )
    lines.append("\\midrule")

    for ds_idx, dataset in enumerate(datasets):
        display = DISPLAY_NAMES.get(dataset, dataset.capitalize())
        # The clean-test likelihood PC value is K-independent; take it once.
        clean_likelihood = results[dataset][h_values[0]]["test"]["likelihood"]

        for row_idx, k in enumerate(h_values):
            cells: list[str] = []
            if row_idx == 0:
                cells.append(f"\\multirow{{{len(h_values)}}}{{*}}{{{display}}}")
            else:
                cells.append("")
            cells.append(str(k))

            for split in SPLITS:
                values = results[dataset][k][split]
                split_cells = render_split_cells(values, precision)
                if split == "test":
                    # Merge the clean-test likelihood PC cell across all K rows.
                    if row_idx == 0:
                        best, second = rank_models(values)
                        merged = format_cell(
                            clean_likelihood,
                            "likelihood" in best,
                            "likelihood" in second,
                            precision,
                        )
                        split_cells[0] = f"\\multirow{{{len(h_values)}}}{{*}}{{{merged}}}"
                    else:
                        split_cells[0] = ""
                cells.extend(split_cells)

            lines.append(" & ".join(cells) + " \\\\")

        if ds_idx < len(datasets) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="DEBD dataset names (default: the 5 from the reference table).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("table.txt"),
        help="Write LaTeX to this file (default: table.txt).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal places for each value (default: 2).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optionally write CSV results with columns dataset,h,method,T,T_a,T_r.",
    )
    args = parser.parse_args()

    results = collect_results(args.datasets)
    latex = render_latex(results, args.datasets, args.precision)

    args.output.write_text(latex + "\n")
    print(f"Wrote LaTeX table to {args.output}")
    if args.csv_output is not None:
        write_csv(results, args.datasets, args.csv_output)
        print(f"Wrote CSV table to {args.csv_output}")


if __name__ == "__main__":
    main()
