import os
import time
import torch
import torch.distributed as dist

def main():
    """Stress-mimic the checkpoint-save pattern that wedged: many mixed-size
    collectives (all_gather bursts like FSDP FULL_STATE_DICT) interleaved with
    barriers, repeated. A LL-protocol race wedges within a few rounds."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(rank)
    t0 = time.time()
    for round_i in range(30):
        # burst of small all_gathers (per-layer FSDP gathers)
        for _ in range(20):
            x = torch.randn(4096, 512, device=f"cuda:{rank}")
            out = [torch.empty_like(x) for _ in range(ws)]
            dist.all_gather(out, x)
        # one big gather (embedding-sized tensor)
        big = torch.randn(64 * 1024 * 1024, device=f"cuda:{rank}")  # 256MB
        outb = [torch.empty_like(big) for _ in range(ws)]
        dist.all_gather(outb, big)
        # tiny all_reduce + barrier (end-of-save pattern)
        y = torch.ones(1, device=f"cuda:{rank}")
        dist.all_reduce(y)
        dist.barrier()
        torch.cuda.synchronize()
        if rank == 0 and round_i % 10 == 0:
            print(f"round {round_i} ok at {time.time()-t0:.1f}s", flush=True)
        del big, outb
    torch.cuda.synchronize()
    if rank == 0:
        print(f"STRESS OK: 30 rounds in {time.time()-t0:.1f}s", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
