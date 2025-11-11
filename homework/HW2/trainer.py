# train.py
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from thop import profile




def prepare(batch, device):
    *inputs, target = batch
    inputs = [x.to(device) for x in inputs]
    return inputs, target.to(device)


def build_loaders(dataset, batch_size, val_split=0.2):
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


def calc_loss(
    logits,
    targets,
    eop_idx,  
    eos_idx 
):
  
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

    
    loss_per_token = F.cross_entropy(
        logits.transpose(1, 2),  
        targets,
        reduction="none"
    )

    denom = mask.sum().clamp(min=1)   # avoid division by zero
    loss = (loss_per_token * mask).sum() / denom

    return loss, denom



def _evaluate(
    model,
    loader,
    eop_idx,
    end_idx,
    device,
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
    return total_loss / max(n_tokens, 1)


def train(
    model,
    dataset,
    num_epochs,
    optimizer,
    scheduler,
    device,
    batch_size
) -> Dict[str, List[Any]]:

    
    model.to(device).train()
    train_loader, val_loader = build_loaders(dataset, batch_size)

    # FLOPs estimate on one minibatch (gracefully fallback if thop unsupported)
    dummy_batch = next(iter(train_loader))
    dummy_inputs, _ = prepare(dummy_batch, device)
    flops_per_step, _ = profile(
        model, inputs=tuple(dummy_inputs), verbose=False
    )

    hist = {'train_loss': [], 'val_loss': [], 'flops': [], 'tokens': []}
    cum_flops = 0
    cum_tokens = 0

    for epoch in range(1, num_epochs + 1):

        train_loss = 0.0
        n_tokens = 0
        
        for batch in train_loader:
            inputs, target = prepare(batch, device)

            optimizer.zero_grad()
            logits, _ = model(*inputs)
            loss, t = calc_loss(logits, target, dataset.eop_idx, dataset.end_idx)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            cum_flops += flops_per_step
            cum_tokens += t

            train_loss += loss.item() * t 
            n_tokens += t



        scheduler.step()
        val_loss = _evaluate(model, val_loader, dataset.eop_idx, dataset.end_idx, device)
        print(f"Epoch {epoch} | Train loss: {(train_loss / n_tokens):.6f} | Val loss {val_loss:.6f}")


        hist["train_loss"].append((train_loss / n_tokens).item())
        hist["flops"].append(int(cum_flops))
        hist["val_loss"].append(val_loss.item())
        hist["tokens"].append(cum_tokens.item())
    return hist