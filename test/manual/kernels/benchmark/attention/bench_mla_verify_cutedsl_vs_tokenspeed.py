"""Kimi K3 MLA multi-token decode (target verify), fp8 KV cache: FlashInfer cute-dsl vs
tokenspeed_mla, one GPU, per layer.

  mode dcp8: per-rank kernel of TP8 + DCP8 (96 query heads, this rank's 1/8 shard of
             every context, global causal lengths passed as each kernel's DCP input)
  mode tp8 : TP8 without DCP (12 heads over the full context)

Lengths: "uniform" = every request at L; "mixed" = per-request global lengths drawn
uniformly from [0.1 L, L] (seeded by L and batch size, identical for both kernels).
Cold L2: do_bench rotates q and flushes L2 before each graph replay; within a replay,
successive calls cycle through block tables over disjoint KV pages so that the KV they
read adds up to at least 4x L2. Times are the median over replays. Under DCP both kernels
also return the base-2 log-sum-exp the cross-rank merge consumes, as the tokenspeed_mla
backend requests it.

usage: python bench_mla_verify_cutedsl_vs_tokenspeed.py <dcp8|tp8> <L in tokens>
           [--batches 1,8,16] [--q-lens 4,8] [--max-kv-mib N]
Prints one CSV row per case: RESULT,mode,L,lengths,q_len,batch,mean_global_len,kv_mib,
cutedsl_us,tokenspeed_us,tokenspeed_over_cutedsl,output_cos"""

import argparse
import math

import flashinfer.decode as fid
import tokenspeed_mla
import torch

from sglang.kernels.jit.benchmark.marker import do_bench

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("dcp8", "tp8"))
parser.add_argument("length", type=int)
parser.add_argument("--batches", default="1,2,4,8,12,16,24,32,40,48,64,96,128")
parser.add_argument("--q-lens", default="4,8")
parser.add_argument(
    "--max-kv-mib", type=float, default=None, help="skip cases reading more KV"
)
args = parser.parse_args()

MODE, L = args.mode, args.length
BATCHES = tuple(int(b) for b in args.batches.split(","))
Q_LENS = tuple(int(q) for q in args.q_lens.split(","))
DCP = 8 if MODE == "dcp8" else 1
HEADS = 96 if MODE == "dcp8" else 12
PAGE, KV_LORA, ROPE, NOPE = 64, 512, 64, 128
SCALE = (NOPE + ROPE) ** -0.5
dev = torch.device("cuda")


def local_len(g):
    return -(-g // DCP)


PAGE_BYTES = PAGE * (KV_LORA + ROPE)
COLD_BYTES = 4 * torch.cuda.get_device_properties(dev).L2_cache_size
# One fp8 pool shared by all cases: the largest case, plus room for small cases to rotate
# over 4x L2 of distinct pages. Filled in chunks so the random-byte temporaries stay small.
max_pages = max(BATCHES) * math.ceil(local_len(L) / PAGE) + COLD_BYTES // PAGE_BYTES + 8
gen = torch.Generator(device=dev).manual_seed(0)
kv_bytes = torch.empty(
    max_pages, 1, PAGE, KV_LORA + ROPE, device=dev, dtype=torch.uint8
)
for start in range(0, max_pages, 4096):
    chunk = kv_bytes[start : start + 4096]
    mag = torch.randint(
        0x30, 0x48, chunk.shape, generator=gen, device=dev, dtype=torch.uint8
    )
    sign = torch.randint(
        0, 2, chunk.shape, generator=gen, device=dev, dtype=torch.uint8
    )
    chunk.copy_(mag | (sign << 7))
del mag, sign
kv = kv_bytes.view(torch.float8_e4m3fn)
perm = torch.randperm(max_pages - 8, generator=gen, device=dev).to(torch.int32)
fi_ws = torch.zeros(512 << 20, dtype=torch.int8, device=dev)
ts_ws = torch.empty(
    tokenspeed_mla.get_num_sm(dev) * HEADS * max(8, *Q_LENS) * (KV_LORA + 1) * 4,
    dtype=torch.int8,
    device=dev,
)


def case(b, kind):
    """Global and local lengths, and block tables over disjoint pages (one per rotation)."""
    g = torch.Generator().manual_seed(L * 131 + b)
    if kind == "uniform":
        glob = torch.full((b,), L, dtype=torch.int32)
    else:
        glob = torch.randint(int(0.1 * L), L + 1, (b,), generator=g, dtype=torch.int32)
    loc = torch.tensor([local_len(int(x)) for x in glob], dtype=torch.int32)
    pages = [math.ceil(int(x) / PAGE) for x in loc]
    # do_bench captures >= 100 calls per replay; more tables than that are never read.
    num_tables = min(
        math.ceil(COLD_BYTES / (sum(pages) * PAGE_BYTES)),
        len(perm) // sum(pages),
        128,
    )
    # cute-dsl needs the table width in whole 128-token tiles (SGLang pads the same way).
    width = -(-max(pages) // (128 // PAGE)) * (128 // PAGE)
    tables = torch.zeros(num_tables, b, width, dtype=torch.int32)
    start = 0
    for t in range(num_tables):
        for i, p in enumerate(pages):
            tables[t, i, :p] = perm[start : start + p].cpu()
            start += p
    # One allocation per table: cute-dsl needs each table's start 16-byte aligned.
    return glob.to(dev), loc.to(dev), [t.to(dev) for t in tables], int(loc.max())


def rotating(fn, glob, loc, tables, max_loc):
    """fn over the next block table on every call, so a captured loop reads cold KV."""
    calls = iter(range(1 << 62))
    return lambda q: fn(q, glob, loc, tables[next(calls) % len(tables)], max_loc)


def cute(q, glob, loc, tables, max_loc):
    extra = {}
    if DCP > 1:
        extra = dict(
            enable_dcp=True,
            cp_world=DCP,
            cp_rank=0,
            causal_seqlens_kv_global=glob,
            return_lse=True,
            return_lse_base="base2",
        )
    return fid.trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv,
        workspace_buffer=fi_ws,
        qk_nope_head_dim=NOPE,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=tables,
        seq_lens=loc,
        max_seq_len=max_loc,
        bmm1_scale=SCALE,
        backend="cute-dsl",
        enable_pdl=True,
        **extra,
    )


def ts(q, glob, loc, tables, max_loc):
    extra = (
        dict(causal_seqs=glob, cp_world=DCP, cp_rank=0, return_lse=True)
        if DCP > 1
        else {}
    )
    return tokenspeed_mla.tokenspeed_mla_decode(
        query=q,
        kv_cache=kv,
        workspace_buffer=ts_ws,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=tables,
        seq_lens=loc,
        max_seq_len=max_loc,
        softmax_scale=SCALE,
        output_scale=1.0,
        enable_pdl=True,
        **extra,
    )


for q_len in Q_LENS:
    for kind in ("uniform", "mixed"):
        for b in BATCHES:
            glob, loc, tables, max_loc = case(b, kind)
            kv_mib = int(loc.sum()) * (KV_LORA + ROPE) / 2**20
            if args.max_kv_mib is not None and kv_mib > args.max_kv_mib:
                continue
            q = (
                torch.randn(b, q_len, HEADS, KV_LORA + ROPE, device=dev, generator=gen)
                * 0.5
            ).to(torch.float8_e4m3fn)
            res, outs = {}, {}
            for name, fn in (("cute-dsl", cute), ("tokenspeed", ts)):
                try:
                    out = fn(q, glob, loc, tables[0], max_loc)
                    out = out[0] if isinstance(out, tuple) else out
                    outs[name] = out.float().reshape(b * q_len, HEADS, -1)
                    iters = 1000 if kv_mib < 1024 else 100
                    run = rotating(fn, glob, loc, tables, max_loc)
                    res[name] = (
                        do_bench(
                            run, input_args=(q,), replay_iters=iters, metrics=(0.5,)
                        ).times[0]
                        * 1e6
                    )
                except Exception as e:  # noqa: BLE001 - report unsupported cases
                    res[name] = None
                    print(
                        f"FAIL {MODE} L={L} {kind} q={q_len} b={b} {name}: "
                        f"{type(e).__name__}: {str(e)[:160]}",
                        flush=True,
                    )
            cos = float("nan")
            if len(outs) == 2:
                a, c = outs["cute-dsl"].flatten(), outs["tokenspeed"].flatten()
                cos = torch.nn.functional.cosine_similarity(a, c, dim=0).item()
            ct, tt = res.get("cute-dsl"), res.get("tokenspeed")
            ratio = f"{tt / ct:.3f}" if ct and tt else "nan"
            print(
                f"RESULT,{MODE},{L},{kind},{q_len},{b},{float(glob.float().mean()):.0f},"
                f"{kv_mib:.0f},{ct and round(ct, 2)},{tt and round(tt, 2)},{ratio},"
                f"{cos:.6f}",
                flush=True,
            )
