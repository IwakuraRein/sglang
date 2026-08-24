"""SM120/SM121 ModelOpt W4A16 NVFP4 FlashInfer CuTe DSL tests."""

import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4A16LinearMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import quantize_nvfp4_shard
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


@contextmanager
def _cutedsl_selected():
    backend = Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL
    with mock.patch.object(fp4_utils, "FP4_GEMM_RUNNER_BACKEND", backend):
        yield


def _checkpoint_values(n: int, k: int):
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
    packed, scale, global_scale, dequant = quantize_nvfp4_shard(weight)
    return packed, scale, global_scale, dequant


def _copy(parameter: torch.Tensor, value: torch.Tensor) -> None:
    with torch.no_grad():
        parameter.copy_(value)


def _modelopt_layer(n: int, k: int, widths: list[int]):
    method = ModelOptNvFp4A16LinearMethod(
        ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
    )
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=k,
        output_partition_sizes=widths,
        input_size=k,
        output_size=n,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
    )
    layer.cuda()
    packed, scale, global_scale, dequant = _checkpoint_values(n, k)
    _copy(layer.weight, packed)
    _copy(layer.weight_scale, scale)
    _copy(layer.weight_scale_2, (1 / global_scale).expand(len(widths)))
    return method, layer, dequant


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "SM120/SM121 required",
)
class TestModelOptW4A16NvFp4FlashInfer(CustomTestCase):
    def _run_shape(self, n, k, widths):
        torch.manual_seed(7)
        method, layer, dequant = _modelopt_layer(n, k, widths)
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) / 20

        with (
            mock.patch.object(
                fp4_utils,
                "mm_bf16_fp4",
                wraps=fp4_utils.mm_bf16_fp4,
            ) as mm_bf16_fp4,
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "prepare_nvfp4_layer_for_marlin",
                side_effect=AssertionError("Marlin preparation must not run"),
            ),
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.apply_fp4_marlin_linear",
                side_effect=AssertionError("Marlin GEMM must not run"),
            ),
        ):
            method.process_weights_after_loading(layer)
            self.assertTrue(layer.flashinfer_w4a16_prepared)
            self.assertEqual(layer.flashinfer_w4a16_padded_output_size % 64, 0)

            for m in (1, 64):
                x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                out = method.apply(layer, x, bias)
                reference = (x.float() @ dequant.T + bias.float()).to(torch.bfloat16)
                torch.testing.assert_close(out, reference, rtol=5e-2, atol=5e-2)

            self.assertGreaterEqual(mm_bf16_fp4.call_count, 2)

        # W4A16 keeps activations in BF16; it has no activation-FP4 state.
        self.assertFalse(hasattr(layer, "input_global_scale"))
        self.assertFalse(hasattr(layer, "input_scale_inv"))

    def test_aligned_and_padded(self):
        with _cutedsl_selected():
            self._run_shape(128, 256, [128])
            self._run_shape(160, 336, [80, 80])

    def test_cuda_graph_capture_and_replay(self):
        with _cutedsl_selected():
            method, layer, _ = _modelopt_layer(160, 336, [80, 80])
            method.process_weights_after_loading(layer)
            static_x = torch.randn((4, 336), device="cuda", dtype=torch.bfloat16) / 10

            # Compile/autotune before capture.
            eager = method.apply(layer, static_x).clone()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = method.apply(layer, static_x)
            static_x.copy_(static_x * 0.5)
            expected = method.apply(layer, static_x).clone()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_out, expected, rtol=5e-2, atol=5e-2)
            self.assertFalse(torch.equal(graph_out, eager))


if __name__ == "__main__":
    unittest.main()
