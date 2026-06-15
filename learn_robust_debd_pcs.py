"""Learn adversarially robust HCLT block-size-4 PCs on all DEBD datasets (seed=0).

Mirrors ``learn_debd_pcs.py`` but performs live adversarial (DRO) training: each
EM batch is greedily bit-flip corrupted against the model currently being trained
via the batched ``greedy_corrupt(pc, batch, k)`` below (a torch reimplementation
of the per-row CPU search in ``experiments/adversarial_dataset_generation.py``).
Resulting circuits are saved under ``drospn_learned_pcs/`` with a ``K{k}`` folder.
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

# pyjuice calls random.randint(0, 1e8) when compiling Triton kernels; Py>=3.12 rejects float bounds
_randint = random.randint
random.randint = lambda a, b: _randint(int(a), int(b))

import numpy as np
import pyjuice as juice
import torch
from torch.utils.data import DataLoader, TensorDataset

from fastcircuits import CircuitSerializer, PyjuiceBuilder

DEBD_DIR = Path("DEBD/datasets")
OUT_DIR = Path("drospn_learned_pcs")
BLOCK_SIZE = 4
SEED = 0
EPOCHS = 1000
BATCH_SIZE = 512

# Cap the number of candidate rows materialized per corruption forward pass so
# greedy_corrupt stays within GPU memory even for high-dimensional datasets.
MAX_CAND_ROWS = 1 << 16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(dataset: str, split: str) -> np.ndarray:
    path = DEBD_DIR / dataset / f"{dataset}.{split}.data"
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.float32)


def debd_datasets() -> list[str]:
    return sorted(
        d.name
        for d in DEBD_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.train.data").is_file()
    )


def out_path(dataset: str, k: int) -> Path:
    return (
        OUT_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


@torch.no_grad()
def greedy_corrupt(pc, batch: torch.Tensor, k: int) -> torch.Tensor:
    """Batched greedy bit-flip corruption of ``batch`` against ``pc``.

    Torch port of the per-row search in adversarial_dataset_generation.py: for
    each of ``k`` steps, find (per row) the single bit flip that most decreases
    the log-likelihood under ``pc`` and apply it. Faithfully reproduces the
    reference quirk that a bit is *always* flipped each step, defaulting to bit 0
    when no flip lowers the LL.
    """
    batch = batch.clone()
    n, d = batch.shape
    rows = torch.arange(n, device=batch.device)
    cols_chunk = max(1, MAX_CAND_ROWS // max(1, n))

    for _ in range(k):
        current_ll = pc(batch)
        cand_ll = torch.empty((n, d), device=batch.device, dtype=current_ll.dtype)

        for c0 in range(0, d, cols_chunk):
            c1 = min(c0 + cols_chunk, d)
            c = c1 - c0
            rep = batch.unsqueeze(1).expand(n, c, d).clone()
            cols = torch.arange(c0, c1, device=batch.device)
            rep[:, torch.arange(c, device=batch.device), cols] ^= 1
            lls = pc(rep.reshape(n * c, d))
            cand_ll[:, c0:c1] = lls.reshape(n, c)

        min_ll, best_i = cand_ll.min(dim=1)
        accept = min_ll < current_ll
        best_i = torch.where(accept, best_i, torch.zeros_like(best_i))
        batch[rows, best_i] ^= 1

    return batch


def learn_and_save(dataset: str, k: int, device: torch.device) -> None:
    dst = out_path(dataset, k)
    if dst.exists():
        print(f"Skipping {dataset} ({dst} exists)")
        return

    seed_everything(SEED)
    train_data = torch.from_numpy(load_split(dataset, "train")).to(device).int()
    valid_data = torch.from_numpy(load_split(dataset, "valid")).to(device).int()

    train_batch = min(BATCH_SIZE, len(train_data))
    valid_batch = min(BATCH_SIZE, len(valid_data))
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=train_batch,
        shuffle=False,
        drop_last=len(train_data) > train_batch,
    )
    valid_loader = DataLoader(
        TensorDataset(valid_data),
        batch_size=valid_batch,
        shuffle=False,
        drop_last=False,
    )

    ns = juice.structures.HCLT(train_data, num_latents=BLOCK_SIZE, input_node_params={"num_cats": 2})
    pc = juice.compile(ns)
    pc.to(device)

    optimizer = juice.optim.CircuitOptimizer(pc, lr=0.1, pseudocount=0.1, method="EM")
    scheduler = juice.optim.CircuitScheduler(
        optimizer,
        method="multi_linear",
        lrs=[0.9, 0.1, 0.05],
        milestone_steps=[0, len(train_loader) * 100, len(train_loader) * 500],
    )

    for batch in train_loader:
        x = batch[0].to(device)
        pc(x, record_cudagraph=True).mean().backward()
        break

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            x_adv = greedy_corrupt(pc, x, k)
            optimizer.zero_grad()
            lls = pc(x_adv)
            lls.mean().backward()
            train_ll += lls.mean().item()
            optimizer.step()
            scheduler.step()
        train_ll /= len(train_loader)

        t1 = time.time()
        valid_ll = 0.0
        for batch in valid_loader:
            x = batch[0].to(device)
            valid_ll += pc(x).mean().item()
        valid_ll /= len(valid_loader)
        t2 = time.time()

        print(
            f"[{dataset} K{k} epoch {epoch}/{EPOCHS}] "
            f"adv train LL {train_ll:.2f}, valid LL {valid_ll:.2f} "
            f"({t1 - t0:.1f}s train, {t2 - t1:.1f}s valid)"
        )

    pc.update_parameters()
    circuit = PyjuiceBuilder(block_size=BLOCK_SIZE).build(pc)

    dst.parent.mkdir(parents=True, exist_ok=True)
    CircuitSerializer.save(circuit, dst)
    print(f"Saved {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn adversarially robust HCLT PCs on DEBD datasets."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of greedy bit-flip steps per batch (corruption strength).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="DEBD dataset names (default: all with train splits).",
    )
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be at least 1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = args.datasets if args.datasets else debd_datasets()
    print(f"Found {len(datasets)} DEBD datasets on {device} (K={args.k})")
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        learn_and_save(dataset, args.k, device)


if __name__ == "__main__":
    main()