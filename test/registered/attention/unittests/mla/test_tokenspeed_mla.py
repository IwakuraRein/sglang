import importlib.util
import math
import unittest

import torch

from sglang.srt.layers.attention.tokenspeed_mla_backend import (
    TokenspeedMLABackend,
    _cute_dsl_mla_decode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    MLAAttentionCase,
    run_mla_attention_case,
)
from sglang.test.test_utils import CustomTestCase

# tokenspeed_mla is a CuTe DSL backend for Blackwell (SM100). It additionally
# enforces:
#   - kv_cache_dtype == torch.float8_e4m3fn (kv_cache_dtype=fp8_e4m3)
#   - page_size in {32, 64}
# See python/sglang/srt/layers/attention/tokenspeed_mla_backend.py and
# is_tokenspeed_mla_available() in python/sglang/srt/utils/common.py.
#
# The shared MLAAttentionCase fixture now supports `fp8_kv_cache=True`:
# `MockMLAModelRunner` decouples `kv_cache_dtype` from the model `dtype`
# and routes K writes through the FP8 quantize path. The reference still
# computes against BF16 K (independent of the cache bytes) and tolerates
# FP8 quant noise via a looser tolerance.
_MIN_SM = 100


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if importlib.util.find_spec("tokenspeed_mla") is None:
        return False, "tokenspeed_mla python package is not installed"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < _MIN_SM:
        return (
            False,
            f"tokenspeed_mla requires SM {_MIN_SM // 10}.{_MIN_SM % 10}+ (Blackwell), "
            f"got SM {major}.{minor}",
        )
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


MLA_SHAPE_KWARGS = dict(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    hidden_size=1024,
    max_context_len=256,
)


from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTokenspeedMLAAttentionBackendCorrectness(CustomTestCase):
    # tokenspeed_mla allows page_size in {32, 64} (server_args.py:2809-2813)
    # and requires kv_cache_dtype==fp8_e4m3 (server_args.py:2814-2818).
    # Cover both page sizes, with extend + decode + ragged + page-boundary.
    CASES = (
        # ----- page_size=64 -----
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_exact_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(64,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_below_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(63,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_above_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0,),
            extend_lens=(65,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_prefix_exact_page_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(64,),
            extend_lens=(4,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_cross_page_boundary_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(63,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_ragged_page_boundary_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=64,
            prefix_lens=(0, 32, 64),
            extend_lens=(63, 32, 1),
        ),
        MLAAttentionCase(
            name="mla_decode_tokenspeed_page_boundary_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(62, 63, 64),
        ),
        MLAAttentionCase(
            name="mla_decode_tokenspeed_bsz1_nonzero_prefix_64",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=64,
            prefix_lens=(31,),
        ),
        # ----- page_size=32 -----
        MLAAttentionCase(
            name="mla_extend_tokenspeed_zero_prefix_exact_page_32",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(0,),
            extend_lens=(32,),
        ),
        MLAAttentionCase(
            name="mla_extend_tokenspeed_cross_page_boundary_32",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
        ),
        MLAAttentionCase(
            name="mla_decode_tokenspeed_page_boundary_32",
            backend="tokenspeed_mla",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            page_size=32,
            prefix_lens=(30, 31, 32),
        ),
    )

    def test_projected_mla_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                # Looser tolerance to absorb FP8 quant noise (the reference
                # reads BF16 K independent of the FP8 cache, so per-element
                # drift from the BF16->FP8 cast accumulates through the
                # attention reduction).
                run_mla_attention_case(
                    self,
                    case,
                    fp8_kv_cache=True,
                    atol=2e-1,
                    rtol=2e-1,
                    **MLA_SHAPE_KWARGS,
                )


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestTokenspeedMLACuteDslDecode(CustomTestCase):
    """The cute-dsl decode kernel tokenspeed_mla selects must return what the DCP merge
    expects: scaled attention over this rank's causally visible keys, its log-sum-exp in
    base 2, and the neutral state (out 0, lse -inf) for a query row that sees no key."""

    PAGE = 64
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_NOPE_HEAD_DIM = 128
    K_SCALE = 0.5

    def _inputs(self, *, num_heads, q_len, global_lens, cp_world, cp_rank):
        dev = torch.device("cuda")
        gen = torch.Generator(device=dev).manual_seed(0)
        # DCP rank r holds global tokens r, r + cp_world, r + 2 * cp_world, ...
        local_lens = [g // cp_world + (cp_rank < g % cp_world) for g in global_lens]
        pages = [-(-n // self.PAGE) for n in local_lens]
        # cute-dsl reads the table in whole 128-token tiles.
        width = -(-max(pages) // (128 // self.PAGE)) * (128 // self.PAGE)
        tables = torch.zeros(len(pages), width, dtype=torch.int32, device=dev)
        perm = torch.randperm(sum(pages), generator=gen, device=dev).to(torch.int32)
        start = 0
        for i, n in enumerate(pages):
            tables[i, :n] = perm[start : start + n]
            start += n
        dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM
        kv_cache = (
            torch.randn(sum(pages), 1, self.PAGE, dim, generator=gen, device=dev) * 0.5
        ).to(torch.float8_e4m3fn)
        query = (
            torch.randn(len(pages), q_len, num_heads, dim, generator=gen, device=dev)
            * 0.5
        ).to(torch.float8_e4m3fn)
        return dict(
            query=query,
            kv_cache=kv_cache,
            block_tables=tables,
            seq_lens=torch.tensor(local_lens, dtype=torch.int32, device=dev),
            max_seq_len=max(local_lens),
            softmax_scale=(self.QK_NOPE_HEAD_DIM + self.QK_ROPE_HEAD_DIM) ** -0.5
            * self.K_SCALE,
        )

    def _reference(self, inputs, *, global_lens, cp_world, cp_rank):
        q = inputs["query"].float()
        kv = inputs["kv_cache"].float()
        batch, q_len, heads, _ = q.shape
        out = torch.zeros(batch, q_len, heads, self.KV_LORA_RANK, device=q.device)
        lse = torch.full((batch, q_len, heads), float("-inf"), device=q.device)
        for i, global_len in enumerate(global_lens):
            local_len = int(inputs["seq_lens"][i])
            pages = inputs["block_tables"][i, : -(-local_len // self.PAGE)].long()
            keys = kv[pages, 0].reshape(-1, kv.shape[-1])[:local_len]
            key_pos = torch.arange(local_len, device=q.device) * cp_world + cp_rank
            for t in range(q_len):
                visible = key_pos <= global_len - q_len + t
                if not visible.any():
                    continue
                scores = (q[i, t] @ keys[visible].T) * inputs["softmax_scale"]
                probs = torch.softmax(scores, dim=-1)
                out[i, t] = probs @ keys[visible, : self.KV_LORA_RANK] * self.K_SCALE
                lse[i, t] = torch.logsumexp(scores, dim=-1) / math.log(2)
        return out, lse

    def test_matches_reference(self):
        for name, kwargs in (
            # TP8 + DCP8 target verify: gathered heads over rank 3's KV shard. Rank 3
            # owns the newest key of the first request and the third-newest of the
            # second, which the causal bound hides from the earlier query tokens; in
            # the third, query token 0 sees none of rank 3's single key.
            (
                "dcp8_verify",
                dict(
                    num_heads=96,
                    q_len=4,
                    global_lens=(64 * 8 * 3 + 4, 902, 6),
                    cp_world=8,
                    cp_rank=3,
                ),
            ),
            # TP8 target verify over the full context.
            (
                "tp8_verify",
                dict(
                    num_heads=12,
                    q_len=8,
                    global_lens=(700, 129, 64),
                    cp_world=1,
                    cp_rank=0,
                ),
            ),
        ):
            with self.subTest(case=name):
                inputs = self._inputs(**kwargs)
                cp_world, cp_rank = kwargs["cp_world"], kwargs["cp_rank"]
                out, lse = _cute_dsl_mla_decode(
                    workspace_buffer=torch.zeros(
                        128 << 20, dtype=torch.int8, device="cuda"
                    ),
                    qk_nope_head_dim=self.QK_NOPE_HEAD_DIM,
                    kv_lora_rank=self.KV_LORA_RANK,
                    qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
                    output_scale=self.K_SCALE,
                    return_lse=True,
                    **inputs,
                    **(
                        dict(
                            causal_seqs=torch.tensor(
                                kwargs["global_lens"], dtype=torch.int32, device="cuda"
                            ),
                            cp_world=cp_world,
                            cp_rank=cp_rank,
                        )
                        if cp_world > 1
                        else {}
                    ),
                )
                ref_out, ref_lse = self._reference(
                    inputs,
                    global_lens=kwargs["global_lens"],
                    cp_world=cp_world,
                    cp_rank=cp_rank,
                )
                torch.testing.assert_close(
                    out.float().reshape(ref_out.shape), ref_out, atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    lse.float().reshape(ref_lse.shape), ref_lse, atol=1e-3, rtol=1e-3
                )


class TestTokenspeedMLANoPEPrefillQuantize(CustomTestCase):
    """NoPE MLA layers must still get packed FP8 prefill Q/K.

    Layers built with ``skip_rope`` carry no rotary embedding, so the prefill
    quantize runs with ``cos_sin_cache=None``. It must return the same packed
    ``[nope | pe]`` FP8 layout as the RoPE path, and the head-0 pe slice it
    writes to the KV cache must equal the plain FP8 cast of the unroped
    ``k_pe`` that the decode path reads back.
    """

    T = 7
    NUM_HEADS = 4
    QK_NOPE_HEAD_DIM = 128
    QK_ROPE_HEAD_DIM = 64
    KV_LORA_RANK = 512

    def test_nope_prefill_quantize_packs_without_rope(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        head_dim = self.QK_NOPE_HEAD_DIM + self.QK_ROPE_HEAD_DIM

        q_nope = torch.randn(
            self.T,
            self.NUM_HEADS,
            self.QK_NOPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        q_pe = torch.randn(
            self.T,
            self.NUM_HEADS,
            self.QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        k_nope = torch.randn(
            self.T,
            self.NUM_HEADS,
            self.QK_NOPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        # k_pe reaches the backend as the strided tail of the latent cache.
        latent_cache = torch.randn(
            self.T,
            1,
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        k_pe = latent_cache[:, :, self.KV_LORA_RANK :]

        q_fp8, k_fp8 = TokenspeedMLABackend._fused_rope_fp8_quantize(
            q_nope=q_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            cos_sin_cache=None,
            positions=torch.arange(self.T, device=device),
            is_neox=True,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
        )

        for name, out in (("q", q_fp8), ("k", k_fp8)):
            with self.subTest(tensor=name):
                self.assertEqual(out.shape, (self.T, self.NUM_HEADS, head_dim))
                self.assertEqual(out.dtype, torch.float8_e4m3fn)
                self.assertTrue(out.is_contiguous())

        fp8 = torch.float8_e4m3fn
        nope = self.QK_NOPE_HEAD_DIM
        self.assertTrue(torch.equal(q_fp8[..., :nope], q_nope.to(fp8)))
        self.assertTrue(torch.equal(q_fp8[..., nope:], q_pe.to(fp8)))
        self.assertTrue(torch.equal(k_fp8[..., :nope], k_nope.to(fp8)))
        # The slice prepare_prefill_qkv writes into the KV cache; the decode
        # NoPE path reads it back as a plain cast of the unroped k_pe.
        self.assertTrue(torch.equal(k_fp8[:, 0:1, nope:], k_pe.to(fp8)))


if __name__ == "__main__":
    unittest.main()
