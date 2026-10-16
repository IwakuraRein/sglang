import unittest

import torch

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_EXPERTS, _TOP_K, _LATENT, _INTERMEDIATE = 896, 16, 3584, 384  # TP8 shard


def _trtllm_gen_rows(x, ids, w13, s13, w2, s2):
    """Unweighted per-(token, slot) expert rows from the trtllm-gen SiTU MoE (the
    serving kernel the persistent experts replace), in top-k order."""
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    num_tokens = x.shape[0]
    xq, xs = mxfp8_quantize(x, False)
    ones = torch.ones(num_tokens, _TOP_K, device=x.device, dtype=torch.bfloat16)
    packed = (ids << 16) | (ones.view(torch.int16).to(torch.int32) & 0xFFFF)
    gemm2_out, _, permuted = trtllm_fp4_block_scale_routed_moe(
        topk_ids=packed,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xs.reshape(-1),
        gemm1_weights=w13,
        gemm1_weights_scale=s13.view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm1_alpha=torch.full((_EXPERTS,), 4.0, device=x.device),
        gemm1_beta=torch.full((_EXPERTS,), 25.0, device=x.device),
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=s2.view(torch.float8_e4m3fn),
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=_EXPERTS,
        top_k=_TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=_INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value,
        activation_type=ActivationType.Situ.value,
        tune_max_num_tokens=max(8, 1 << (num_tokens - 1).bit_length()),
        output=torch.empty(num_tokens, _LATENT, device=x.device, dtype=torch.bfloat16),
        do_finalize=False,
    )
    return gemm2_out[permuted.view(-1).long()].view(num_tokens, _TOP_K, _LATENT)


@unittest.skipIf(
    not torch.cuda.is_available() or get_device_sm() // 10 != 10,
    "Kimi K3 persistent experts require SM100/SM103",
)
class TestKimiK3PersistentExperts(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.moe.kimi_k3.experts_runtime import (
            PersistentExperts,
            scratch,
        )

        torch.manual_seed(0)
        device = torch.device("cuda")
        u8 = lambda *shape: torch.randint(
            0, 256, shape, device=device, dtype=torch.uint8
        )
        # Random bytes in the trtllm-gen layout: both kernels must read the same
        # bytes the same way. E8M0 scales near 2^-6 keep the SiTU inputs in range.
        cls.w13 = u8(_EXPERTS, 2 * _INTERMEDIATE, _LATENT // 2)
        cls.w2 = u8(_EXPERTS, _LATENT, _INTERMEDIATE // 2)
        scale = lambda *shape: torch.randint(
            118, 124, shape, device=device, dtype=torch.uint8
        )
        cls.s13 = scale(_EXPERTS, 2 * _INTERMEDIATE, _LATENT // 32)
        cls.s2 = scale(_EXPERTS, _LATENT, _INTERMEDIATE // 32)
        cls.experts = PersistentExperts(cls.w13, cls.s13, cls.w2, cls.s2)
        cls.buffers = scratch(8, device)

    def _check(self, num_tokens):
        x = (torch.randn(num_tokens, _LATENT, device="cuda") * 2).to(torch.bfloat16)
        ids = torch.stack(
            [torch.randperm(_EXPERTS, device="cuda")[:_TOP_K] for _ in range(num_tokens)]
        ).to(torch.int32)
        out = torch.empty(
            num_tokens, _TOP_K, _LATENT, device="cuda", dtype=torch.bfloat16
        )
        self.experts.prepare(num_tokens, self.buffers, out)
        self.experts.run(x, ids)
        ref = _trtllm_gen_rows(x, ids, self.w13, self.s13, self.w2, self.s2).float()
        got = out.float()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), ref.flatten(), dim=0
        ).item()
        rel_rms = ((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()
        # The two kernels round the SiTU hidden to MXFP8 independently (~2% RMS).
        self.assertGreater(cos, 0.999, f"num_tokens={num_tokens}")
        self.assertLess(rel_rms, 0.04, f"num_tokens={num_tokens}")

    def test_small_batches(self):
        for num_tokens in (1, 2, 3, 4, 5, 7, 8):
            self._check(num_tokens)

    def test_grouped_batches(self):
        for num_tokens in (9, 16, 32, 64):
            self._check(num_tokens)

    def test_small_batch_after_grouped_batch(self):
        """A B <= 8 call right after a grouped (B > 8) call on the shared workspace
        must not read the hidden images its predecessor left un-re-armed."""
        for num_tokens in (8, 16, 8, 8, 64, 1):
            self._check(num_tokens)


if __name__ == "__main__":
    unittest.main()
