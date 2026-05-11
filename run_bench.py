import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys, json, time, torch, gc
from config import Config, train_paths
from model import GPT2
from data import *

strategy, bs, nb, flash = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))

config = Config()
config.use_flash_attn = flash
model = GPT2(config).to("cuda")

dl = make_loader(strategy, train_paths, bs, nb)
dl = iter(dl)
warmup = next(dl)
transfer_batch(warmup, "cuda", flash)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(3):
        model(**warmup)
    
    torch.cuda.reset_peak_memory_stats()
    times = []
    total_tokens = 0
    
    for batch in dl:
        transfer_batch(batch, "cuda", flash)
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(**batch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        if "attention_mask" in batch:
            mask = batch["attention_mask"]
            if mask.dim() == 2:
                total_tokens += mask.sum().item()
            else:
                total_tokens += mask.diagonal(dim1=-2, dim2=-1).sum().item()
        elif "cu_seqlens" in batch:
            total_tokens += batch["cu_seqlens"][0][-1].item()
        
        times.append(dict(strategy=strategy, bs=bs, nb=nb, flash=flash,
                          time=end-start, total_tokens=total_tokens))
    
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(json.dumps({"times": times, "peak_mem": peak_mem}))