from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.utils.common import (
    get_device_capability,
    is_cuda,
    is_sm100_supported,
)
from sglang.srt.utils.custom_op import (
    register_custom_op,
    register_custom_op_from_extern,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


fp4_quantize = None
try:
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize

    _flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"

    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def _flashinfer_fp4_quantize_impl(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _flashinfer_fp4_quantize(
            input=input,
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
            backend=_flashinfer_fp4_quantize_backend,
        )

    def _flashinfer_fp4_quantize_fake(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_column_major = input.stride(-2) == 1
        if is_column_major:
            m = input.shape[-1]
            K = input.shape[-2]
        else:
            m = input.numel() // input.shape[-1]
            K = input.shape[-1]
        if is_column_major:
            x_q = input.new_empty((*input.shape[:-2], K // 2, m), dtype=torch.uint8)
        else:
            x_q = input.new_empty((*input.shape[:-1], K // 2), dtype=torch.uint8)
        if is_sf_swizzled_layout:
            row_size = 8 if is_sf_8x4_layout else 128
            sf_rows = _round_up(m, row_size)
            sf_cols = _round_up(K // sf_vec_size, 4)
        else:
            sf_rows = m
            sf_cols = K // sf_vec_size
        if is_column_major:
            sf = input.new_empty((sf_cols, sf_rows), dtype=torch.uint8)
        else:
            sf = input.new_empty((sf_rows, sf_cols), dtype=torch.uint8)
        return x_q, sf

    fp4_quantize = register_custom_op_from_extern(
        _flashinfer_fp4_quantize_impl,
        op_name="flashinfer_fp4_quantize",
        fake_impl=_flashinfer_fp4_quantize_fake,
    )
except ImportError:
    fp4_quantize = None


# Keep the W4A16 imports separate from the W4A4 quantization API above. Some
# FlashInfer builds expose only a subset of these APIs, and W4A4 must remain
# usable in that case.
try:
    from flashinfer import prepare_bf16_fp4_weights
except ImportError:
    prepare_bf16_fp4_weights = None

try:
    from flashinfer import mm_bf16_fp4
except ImportError:
    mm_bf16_fp4 = None

try:
    from flashinfer import nvfp4_block_scale_interleave
except ImportError:
    nvfp4_block_scale_interleave = None


FLASHINFER_W4A16_SUPPORTED_CAPABILITIES = {
    (10, 0),
    (10, 3),
    (11, 0),
    (12, 0),
    (12, 1),
}
FLASHINFER_W4A16_ALIGNMENT = 64


def _flashinfer_w4a16_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    del weight, weight_scale, alpha
    return input.new_empty((input.shape[0], output_size))


@register_custom_op(fake_impl=_flashinfer_w4a16_gemm_fake)
def flashinfer_w4a16_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Graph-safe wrapper around FlashInfer's prepared BF16 x FP4 GEMM."""
    if mm_bf16_fp4 is None:
        raise RuntimeError(
            "W4A16 NVFP4 CuTe DSL GEMM requires flashinfer.mm_bf16_fp4. "
            "Use --fp4-gemm-backend flashinfer_cutedsl --dtype bfloat16."
        )
    return mm_bf16_fp4(
        input,
        weight,
        weight_scale,
        alpha,
        backend="cute-dsl",
        out_dtype=input.dtype,
    )


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    MARLIN = "marlin"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_flashinfer_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutedsl(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL

    def is_marlin(self) -> bool:
        return self == Fp4GemmRunnerBackend.MARLIN

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def get_flashinfer_backend(self) -> str:
        """Get the backend string to pass to FlashInfer's mm_fp4 API.

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        Examples:
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
            'flashinfer_cutedsl' -> 'cute-dsl'
        """
        if self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL:
            return "cute-dsl"
        if self.value.startswith("flashinfer_"):
            return self.value.removeprefix("flashinfer_")
        else:
            return self.value


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None


def initialize_fp4_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP4 GEMM configuration from server args."""
    global FP4_GEMM_RUNNER_BACKEND

    backend = server_args.fp4_gemm_runner_backend
    if backend == "auto":
        if is_sm100_supported():
            backend = "flashinfer_cutedsl"
        elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):
            backend = "marlin"
        else:
            backend = "flashinfer_cutlass"

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:
    """Get the current FP4 GEMM runner backend."""
    global FP4_GEMM_RUNNER_BACKEND
    if FP4_GEMM_RUNNER_BACKEND is None:
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO
    return FP4_GEMM_RUNNER_BACKEND


def validate_flashinfer_w4a16(
    *,
    params_dtype: torch.dtype,
    group_size: int,
    device: torch.device | None = None,
) -> None:
    if params_dtype != torch.bfloat16:
        raise ValueError(
            f"FlashInfer W4A16 NVFP4 requires BF16 activations, got {params_dtype}. "
            "Use --fp4-gemm-backend flashinfer_cutedsl --dtype bfloat16."
        )
    if group_size != 16:
        raise ValueError(
            f"FlashInfer W4A16 NVFP4 requires group_size=16, got {group_size}."
        )
    missing_apis = [
        name
        for name, api in (
            ("prepare_bf16_fp4_weights", prepare_bf16_fp4_weights),
            ("mm_bf16_fp4", mm_bf16_fp4),
            ("nvfp4_block_scale_interleave", nvfp4_block_scale_interleave),
        )
        if api is None
    ]
    if missing_apis:
        raise RuntimeError(
            "FlashInfer W4A16 NVFP4 support is unavailable; missing API(s): "
            f"{', '.join(missing_apis)}. Install a compatible FlashInfer build."
        )

    if device is not None and device.type != "cuda":
        raise RuntimeError(
            "FlashInfer W4A16 NVFP4 requires a supported Blackwell CUDA GPU."
        )
    capability = get_device_capability()
    if capability not in FLASHINFER_W4A16_SUPPORTED_CAPABILITIES:
        raise RuntimeError(
            "FlashInfer W4A16 NVFP4 requires compute capability one of "
            f"{sorted(FLASHINFER_W4A16_SUPPORTED_CAPABILITIES)}, got {capability}."
        )


def _shared_fused_scale(scale: torch.Tensor, scale_name: str) -> torch.Tensor:
    flat_scale = scale.detach().to(torch.float32).reshape(-1)
    if flat_scale.numel() == 0:
        raise ValueError(f"{scale_name} must contain at least one value.")
    if not torch.equal(flat_scale, flat_scale[0].expand_as(flat_scale)):
        raise ValueError(
            "FlashInfer W4A16 NVFP4 requires one shared global scale across "
            f"all fused output shards, but {scale_name} contains "
            f"{flat_scale.tolist()}."
        )
    return flat_scale[:1]


def prepare_w4a16_nvfp4_for_flashinfer(
    layer: torch.nn.Module,
    *,
    weight_name: str,
    weight_scale_name: str,
    global_scale_name: str,
    alpha_name: str,
    group_size: int = 16,
) -> None:
    """Prepare checkpoint-linear NVFP4 weights once for BF16 x FP4 CuTe DSL."""
    if getattr(layer, "flashinfer_w4a16_prepared", False):
        return

    weight = getattr(layer, weight_name)
    weight_scale = getattr(layer, weight_scale_name)
    params_dtype = getattr(layer, "params_dtype", None)
    validate_flashinfer_w4a16(
        params_dtype=params_dtype,
        group_size=group_size,
        device=weight.device,
    )

    logical_output_size = int(layer.output_size_per_partition)
    input_size = int(layer.input_size_per_partition)
    if weight.dtype != torch.uint8 or weight.dim() != 2:
        raise TypeError(
            "FlashInfer W4A16 NVFP4 expects a 2-D uint8 packed weight, got "
            f"shape={tuple(weight.shape)}, dtype={weight.dtype}."
        )
    expected_weight_shape = (logical_output_size, input_size // 2)
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"Packed weight shape {tuple(weight.shape)} does not match "
            f"expected {expected_weight_shape}."
        )
    expected_scale_shape = (logical_output_size, input_size // group_size)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Weight scale shape {tuple(weight_scale.shape)} does not match "
            f"expected {expected_scale_shape}."
        )
    if weight_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "FlashInfer W4A16 NVFP4 expects E4M3 block scales, got "
            f"{weight_scale.dtype}."
        )

    shared_scale = _shared_fused_scale(
        getattr(layer, global_scale_name), global_scale_name
    )
    alpha = shared_scale.to(device=weight.device, dtype=torch.float32).reshape(1)

    padded_output_size = (
        (logical_output_size + FLASHINFER_W4A16_ALIGNMENT - 1)
        // FLASHINFER_W4A16_ALIGNMENT
        * FLASHINFER_W4A16_ALIGNMENT
    )
    pad_rows = padded_output_size - logical_output_size
    if pad_rows:
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows)).contiguous()
        weight_scale = torch.nn.functional.pad(
            weight_scale, (0, 0, 0, pad_rows)
        ).contiguous()
    else:
        weight = weight.contiguous()
        weight_scale = weight_scale.contiguous()

    # FlashInfer's helper takes the canonical 128x4 byte layout emitted by
    # nvfp4_quantize. Checkpoints store the equivalent E4M3 scale matrix in
    # linear [N, K/16] order.
    canonical_weight_scale = nvfp4_block_scale_interleave(
        weight_scale.view(torch.uint8)
    )
    prepared_weight, prepared_scale, prepared_alpha = prepare_bf16_fp4_weights(
        weight,
        canonical_weight_scale,
        alpha,
        backend="cute-dsl",
    )
    if prepared_alpha is None:
        prepared_alpha = alpha

    setattr(layer, weight_name, Parameter(prepared_weight, requires_grad=False))
    setattr(
        layer,
        weight_scale_name,
        Parameter(prepared_scale, requires_grad=False),
    )
    setattr(layer, alpha_name, Parameter(prepared_alpha, requires_grad=False))
    if global_scale_name != alpha_name:
        delattr(layer, global_scale_name)
    layer.flashinfer_w4a16_prepared = True
    layer.flashinfer_w4a16_padded_output_size = padded_output_size


def apply_w4a16_nvfp4_flashinfer(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    weight_name: str,
    weight_scale_name: str,
    alpha_name: str,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise ValueError(
            f"FlashInfer W4A16 NVFP4 requires BF16 activations, got {x.dtype}. "
            "Use --fp4-gemm-backend flashinfer_cutedsl --dtype bfloat16."
        )
    input_2d = x.reshape(-1, x.shape[-1])
    output = flashinfer_w4a16_gemm(
        input_2d,
        getattr(layer, weight_name),
        getattr(layer, weight_scale_name),
        getattr(layer, alpha_name),
        int(layer.flashinfer_w4a16_padded_output_size),
    )
    logical_output_size = int(layer.output_size_per_partition)
    if output.shape[-1] != logical_output_size:
        output = output[:, :logical_output_size].contiguous()
    if bias is not None:
        output = output + bias
    return output.reshape(*x.shape[:-1], logical_output_size)
