"""Kimi K3 decode routed experts on the CuTe DSL persistent kernel.

Replaces the trtllm-gen routed MoE for decode batches of <= 64 tokens. The kernel
reads the trtllm-gen MXFP4 weight layout as loaded, quantizes the latent rows to
MXFP8 itself, and writes unweighted per-(token, slot) rows in top-k order; the
caller hands them to the deferred-finalize all-reduce as an identity permutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

_TOP_K = 16
_NUM_EXPERTS = 896
_LATENT = 3584
_MAX_TOKENS = 64
# (w13 rows, w2 row bytes) per rank: TP8 I=384, TP16 I=256 (padded).
_SUPPORTED_SHAPES = {(768, 192), (512, 128)}

# Workspace, plan and output rows are shared by every MoE layer of the model: the
# layers run in order on one stream and each output is consumed by its own finalize.
_SHARED: dict[torch.device, dict[str, torch.Tensor]] = {}


def is_supported(
    *,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    num_local_experts: int,
    top_k: int,
    situ_betas: tuple[Optional[float], Optional[float]],
) -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(w13.device)[0] != 10:
        return False
    if num_local_experts != _NUM_EXPERTS or top_k != _TOP_K:
        return False
    if situ_betas != (4.0, 25.0):
        return False
    if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
        return False
    if w13.shape[0] != _NUM_EXPERTS or w13.shape[2] != _LATENT // 2:
        return False
    if w2.shape[:2] != (_NUM_EXPERTS, _LATENT):
        return False
    rows, row_bytes = w13.shape[1], w2.shape[2]
    if (rows, row_bytes) not in _SUPPORTED_SHAPES:
        return False
    # MXFP4: one E8M0 byte per 32 weights (NVFP4 would carry one per 16).
    return w13_scale.shape == (_NUM_EXPERTS, rows, _LATENT // 32) and (
        w2_scale.shape == (_NUM_EXPERTS, _LATENT, row_bytes * 2 // 32)
    )


def _shared_buffers(tp: int, device: torch.device) -> dict[str, torch.Tensor]:
    from sglang.kernels.ops.moe.kimi_k3.experts_runtime import scratch

    buffers = _SHARED.get(device)
    if buffers is None:
        buffers = dict(scratch(tp, device))
        buffers["out"] = torch.empty(
            _MAX_TOKENS, _TOP_K, _LATENT, dtype=torch.bfloat16, device=device
        )
        buffers["identity"] = torch.arange(
            _MAX_TOKENS * _TOP_K, dtype=torch.int32, device=device
        )
        _SHARED[device] = buffers
    return buffers


class KimiK3PersistentExperts:
    """One MoE layer's persistent-experts runner over its loaded trtllm-gen weights."""

    def __init__(
        self,
        *,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        beta: float,
        linear_beta: float,
    ):
        from sglang.kernels.ops.moe.kimi_k3.experts_runtime import PersistentExperts

        self._experts = PersistentExperts(
            w13,
            w13_scale.view(torch.uint8),
            w2,
            w2_scale.view(torch.uint8),
            beta=beta,
            linear_beta=linear_beta,
        )
        self._buffers = _shared_buffers(self._experts.tp, w13.device)
        self._prepared: set[int] = set()

    def ready(self, num_tokens: int) -> bool:
        """Whether a batch of ``num_tokens`` runs here (prepared, or preparable now)."""
        if num_tokens in self._prepared:
            return True
        if not 1 <= num_tokens <= _MAX_TOKENS:
            return False
        # Preparing compiles and uploads TMA descriptors, which a graph capture
        # cannot hold; the capture's eager warmup runs prepare each batch first.
        if torch.cuda.is_current_stream_capturing():
            return False
        self._experts.prepare(
            num_tokens, self._buffers, self._buffers["out"][:num_tokens]
        )
        self._prepared.add(num_tokens)
        return True

    def forward_deferred(
        self,
        routed_input: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> FlashInferTrtllmDeferredFinalizeOutput:
        """Deferred-finalize triple for ``routed_input`` [T, 3584]; requires
        ``ready(T)``."""
        # Deferred: flashinfer_trtllm and the quantization package import each other.
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
        )

        num_tokens = routed_input.shape[0]
        x = routed_input.to(torch.bfloat16).contiguous()
        out = self._experts.run(x, topk_ids.to(torch.int32).contiguous())
        rows = num_tokens * _TOP_K
        return FlashInferTrtllmDeferredFinalizeOutput(
            gemm2_out=out.view(rows, _LATENT),
            expert_weights=topk_weights,
            expanded_idx_to_permuted_idx=self._buffers["identity"][:rows],
            top_k=_TOP_K,
        )
