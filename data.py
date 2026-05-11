from config import SKIP_LENGTH, LINE_LIMITS, tokenizer
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from sortedcontainers import SortedList
import math
import random
import torch
from torch.utils.data import Sampler, IterableDataset, DataLoader, Dataset


def get_lines(paths, line_limits):
    lines = []
    for path in paths:
        with open(path) as f:
            for line in f:
                lines.append(line.strip())
                if line_limits and len(lines) >= line_limits:
                    return lines
    return lines

class BaseDataset(Dataset):
    def __init__(self, paths, tokenizer, max_length=SKIP_LENGTH, line_limits=LINE_LIMITS):
        self.lines = get_lines(paths, line_limits)
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

        
class StaticPadDataset(BaseDataset):
    """Pad everything to max_length."""
    def __getitem__(self, idx):
        enc = self.tokenizer(self.lines[idx], truncation=True, max_length=self.max_length,
                        padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


class DynamicPadDataset(BaseDataset):
    """Pad to max in batch via dynamic_collate_fn."""
    def __getitem__(self, idx):
        return self.lines[idx]


class BinnedDynamicPadDataset(BaseDataset):
    """Same as DynamicPadDataset, batching handled by BinSampler."""
    def __getitem__(self, idx):
        return self.lines[idx]


class BinnedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, n_bins=1):
        bin_width = math.ceil(dataset.max_length / n_bins)
        bins = defaultdict(list)
        for idx, line in enumerate(dataset.lines):
            bins[len(line) // bin_width].append(idx)
        self.batches = []
        for bin_indices in bins.values():
            indices = list(bin_indices)
            random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i + batch_size])

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        random.shuffle(self.batches)
        yield from self.batches


class PackedDataset(Dataset):
    """FFD packing into fixed-length bins."""
    def __init__(self, paths, tokenizer, max_length=SKIP_LENGTH, line_limits=LINE_LIMITS, drop_mask=False):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.drop_mask = drop_mask
        
        all_tokens = self.tokenizer(get_lines(paths, line_limits), truncation=False, padding=False)["input_ids"]

        free_spaces = SortedList()
        bins = defaultdict(list)

        for tokens in sorted(all_tokens, key=len, reverse=True):
            if len(tokens) > SKIP_LENGTH:
                continue
            i = free_spaces.bisect_left((len(tokens),))
            if i >= len(free_spaces):
                bid = len(bins)
                bins[bid].append(tokens)
                free_spaces.add((max_length - len(tokens), bid))
            else:
                free, bid = free_spaces.pop(i)
                bins[bid].append(tokens)
                free_spaces.add((free - len(tokens), bid))

        self.packs = []
        for free, bid in free_spaces:
            ids = sum(bins[bid], []) + [tokenizer.pad_token_id] * free  # should i remove it?
            cu = [0]
            for t in bins[bid]:
                cu.append(cu[-1] + len(t))
            self.packs.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "cu_seqlens": torch.tensor(cu, dtype=torch.long),
            })

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, idx):
        pack = self.packs[idx]
        if self.drop_mask:
            return {
            "input_ids": pack["input_ids"],
            "cu_seqlens": pack["cu_seqlens"],
        }
        ml = self.max_length
        mask = torch.zeros(ml, ml, dtype=torch.bool)
        cu = pack["cu_seqlens"]
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            mask[s:e, s:e] = torch.tril(torch.ones(e - s, e - s, dtype=torch.bool))
        return {
            "input_ids": pack["input_ids"],
            "attention_mask": mask,
            "cu_seqlens": pack["cu_seqlens"],
        }

def make_dynamic_collate_fn(tokenizer, max_length):
    def f(batch,):
        return tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return f


def packed_collate_fn(batch):
    stacked_batch = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "cu_seqlens": [b["cu_seqlens"] for b in batch],
    }
    if "attention_mask" in batch[0]:
        stacked_batch["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    return stacked_batch


def transfer_batch(batch, device, flash):
    batch["input_ids"] = batch["input_ids"].to(device)
    if "attention_mask" in batch and not flash:
        batch["attention_mask"] = batch["attention_mask"].to(device)    
    if "cu_seqlens" in batch: # i use them for pos_emb
        batch["cu_seqlens"] = [c.to(device) for c in batch["cu_seqlens"]]



def make_loader(strategy, paths, bs, nb=1):
    dynamic_collate = make_dynamic_collate_fn(tokenizer, SKIP_LENGTH)
    
    if strategy == "static_pad":
        ds = StaticPadDataset(paths, tokenizer)
        return DataLoader(ds, batch_size=bs)
    elif strategy == "dynamic_pad":
        ds = DynamicPadDataset(paths, tokenizer)
        return DataLoader(ds, batch_size=bs, collate_fn=dynamic_collate)
    elif strategy == "binned_pad":
        ds = BinnedDynamicPadDataset(paths, tokenizer)
        return DataLoader(ds, batch_sampler=BinnedBatchSampler(ds, bs, nb), collate_fn=dynamic_collate)
    elif strategy == "packed_flattened_pad":
        ds = PackedDataset(paths, tokenizer, max_length=SKIP_LENGTH * bs)
        return DataLoader(ds, batch_size=1, collate_fn=packed_collate_fn)
    elif strategy == "packed_flattened_pad_flash":
        ds = PackedDataset(paths, tokenizer, max_length=SKIP_LENGTH * bs, drop_mask=True)
        return DataLoader(ds, batch_size=1, collate_fn=packed_collate_fn)        