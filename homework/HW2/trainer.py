# train.py
from __future__ import annotations

import itertools
import typing as t
from typing import Any

import torch
from types_hw2 import TrainerHistDict
from kret_studies.kret_torch.mixin.constants import DEVICE_LITERAL, DEVICE_TORCH_STR
from thop import profile
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from dataset import DecoderDataset, EncoderDecoderDataset


def prepare(batch: t.Tuple[torch.Tensor, ...], device: DEVICE_LITERAL):
    *inputs, target = batch
    inputs = [x.to(device) for x in inputs]
    return inputs, target.to(device)


def build_loaders(
    dataset: EncoderDecoderDataset | DecoderDataset, batch_size: int, val_split=0.2
):
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    collate = dataset.collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def calc_loss(logits: torch.Tensor, targets: torch.Tensor, eop_idx: int, eos_idx: int):

    B, T, V = logits.shape
    device = targets.device

    eos_pos = torch.argmax((targets == eos_idx).to(torch.int32), dim=1)
    has_eos = (targets == eos_idx).any(dim=1)
    eos_pos = torch.where(has_eos, eos_pos, torch.full_like(eos_pos, T - 1))

    eop_pos = torch.argmax((targets == eop_idx).to(torch.int32), dim=1)
    has_eop = (targets == eop_idx).any(dim=1)
    start_pos = torch.where(has_eop, eop_pos + 1, torch.zeros_like(eop_pos))

    positions = torch.arange(T, device=device).unsqueeze(0)
    mask = (positions >= start_pos.unsqueeze(1)) & (positions <= eos_pos.unsqueeze(1))
    mask = mask.to(logits.dtype)

    loss_per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")

    denom = mask.sum().clamp(min=1)  # avoid division by zero
    loss = (loss_per_token * mask).sum() / denom

    return loss, denom


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    eop_idx: int,
    end_idx: int,
    device: DEVICE_LITERAL,
):
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    with torch.no_grad():
        for batch in loader:
            inputs, target = prepare(batch, device)
            logits, _ = model(*inputs)
            loss, tokens = calc_loss(logits, target, eop_idx, end_idx)
            total_loss += (loss * tokens).item()
            n_tokens += tokens
    model.train()
    return float(total_loss / max(n_tokens, 1))


def train(
    model: nn.Module,
    dataset: EncoderDecoderDataset | DecoderDataset,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ExponentialLR,
    device: DEVICE_LITERAL,
    batch_size: int,
) -> TrainerHistDict:

    model.to(device).train()
    train_loader, val_loader = build_loaders(dataset, batch_size)

    # FLOPs estimate on one minibatch (gracefully fallback if thop unsupported)
    dummy_batch = next(iter(train_loader))
    dummy_inputs, _ = prepare(dummy_batch, device)
    out = profile(model, inputs=tuple(dummy_inputs), verbose=False)
    flops_per_step = out[0]
    # print(type(flops_per_step))

    hist: TrainerHistDict = {
        "train_loss": [],
        "val_loss": [],
        "flops": [],
        "tokens": [],
    }
    cum_flops = 0
    cum_tokens = 0

    for epoch in range(1, num_epochs + 1):

        train_loss = 0
        n_tokens = 0

        for batch in train_loader:
            inputs, target = prepare(batch, device)

            optimizer.zero_grad()
            logits, _ = model(*inputs)
            loss, t2 = calc_loss(logits, target, dataset.eop_idx, dataset.end_idx)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            cum_flops += flops_per_step
            cum_tokens += t2

            train_loss += loss.item() * t2
            n_tokens += t2

        scheduler.step()
        val_loss = _evaluate(
            model, val_loader, dataset.eop_idx, dataset.end_idx, device
        )
        print(
            f"Epoch {epoch} | Train loss: {(train_loss / n_tokens):.6f} | Val loss {val_loss:.6f}"
        )

        hist["train_loss"].append(float(train_loss / n_tokens))
        hist["flops"].append(int(cum_flops))
        hist["val_loss"].append(float(val_loss))
        hist["tokens"].append(int(cum_tokens))
    return hist


import pathlib

HW2_DATA_DIR = "/Users/Akseldkw/coding/Columbia/COMS4776-Data/data/homework/HW2"


def save_model_auto(
    model: nn.Module, base_dir: str | pathlib.Path = HW2_DATA_DIR, model_specs: str = ""
):
    """
    Save model.state_dict() into: base_dir / "<ClassName>.pt"
    Returns the full path.
    """
    base = pathlib.Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    filename = f"{model.__class__.__name__}{model_specs}.pt"
    path = base / filename

    torch.save(model.state_dict(), path)
    return path


def load_model_auto(
    model_cls: type[nn.Module],
    base_dir: str | pathlib.Path = HW2_DATA_DIR,
    device: str = DEVICE_TORCH_STR,
    model_specs: str = "",
    *args,
    **kwargs,
):
    base = pathlib.Path(base_dir)
    path = base / f"{model_cls.__name__}{model_specs}.pt"

    model = model_cls(*args, **kwargs)
    print(f"[INFO] Loading model weights from {path} onto {device} device.")

    # 1. Load checkpoint on CPU to avoid MPS float64 issue
    state = torch.load(path, map_location="cpu")

    # 2. Load into the fresh model
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        print("[WARN] Load from disk failed, keeping random initialization...")

    # 3. Move model to the desired device (mps / cuda / cpu)
    model.to(device)
    return model
