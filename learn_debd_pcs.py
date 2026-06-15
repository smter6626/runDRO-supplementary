"""Learn HCLT block-size-4 PCs on all DEBD datasets (seed=0)."""

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
OUT_DIR = Path("learned_pcs")
BLOCK_SIZE = 4
SEED = 0
EPOCHS = 350
BATCH_SIZE = 512


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


def learn_and_save(dataset: str, device: torch.device) -> None:
    out_path = (
        OUT_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )
    if out_path.exists():
        print(f"Skipping {dataset} ({out_path} exists)")
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
        milestone_steps=[0, len(train_loader) * 100, len(train_loader) * 350],
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
            optimizer.zero_grad()
            lls = pc(x)
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
            f"[{dataset} epoch {epoch}/{EPOCHS}] "
            f"train LL {train_ll:.2f}, valid LL {valid_ll:.2f} "
            f"({t1 - t0:.1f}s train, {t2 - t1:.1f}s valid)"
        )

    pc.update_parameters()
    circuit = PyjuiceBuilder(block_size=BLOCK_SIZE).build(pc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    CircuitSerializer.save(circuit, out_path)
    print(f"Saved {out_path}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = debd_datasets()
    print(f"Found {len(datasets)} DEBD datasets on {device}")
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        learn_and_save(dataset, device)


if __name__ == "__main__":
    main()
