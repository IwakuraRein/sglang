"""FlashInfer KDA prefill integration against SGLang's Triton reference."""

from importlib.util import find_spec
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
    (10, 0),
    (10, 3),
):
    pytest.skip("FlashInfer KDA prefill requires B200/B300", allow_module_level=True)

flashinfer = pytest.importorskip("flashinfer")
pytest.importorskip("flashinfer.kda")
if Version(flashinfer.__version__) < Version("0.7.0"):
    pytest.skip(
        "KDA prefill checkpoints require FlashInfer 0.7.0", allow_module_level=True
    )

from sglang.srt.layers.attention.linear.kda_backend import (  # noqa: E402
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (  # noqa: E402
    FlashInferKDAPrefillKernel,
    build_flashinfer_kda_checkpoint_plan,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import (  # noqa: E402
    TritonKDAKernel,
)
from sglang.srt.layers.attention.linear.utils import (  # noqa: E402
    LinearAttnKernelBackend,
)
from sglang.srt.layers.attention.mamba.prefill_track_metadata import (  # noqa: E402
    build_prefill_track_plan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.test.kits.attention_unittest.attention_methods.kda_attention import (  # noqa: E402
    KDAAttentionCase,
    build_kda_attention_fixture,
    run_kda_fixture_eager,
)


def flashinfer_dispatcher():
    return KDAKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        LinearAttnKernelBackend.FLASHINFER,
        LinearAttnKernelBackend.TRITON,
    )


@pytest.mark.parametrize(
    "state_dtype,layout,prefix_len",
    [
        (torch.bfloat16, "dense", 0),
        (torch.float32, "dense", 0),
        (torch.bfloat16, "padded_slots", 0),
        (torch.float32, "strided_gates", 0),
        (torch.bfloat16, "padded_row", 0),
        (torch.float32, "dense", 512),
    ],
    ids=["bf16", "fp32", "padded_slots", "strided_gates", "padded_row", "dcp8"],
)
def test_kda_prefill_checkpoints(state_dtype, layout, prefix_len):
    torch.manual_seed(11 if prefix_len else 7)
    lengths = [578] if prefix_len else [130, 128]
    if layout == "padded_row":
        lengths.append(0)
    heads, dim = (2 if prefix_len else 12), 128
    total_tokens = sum(lengths)

    def tensor(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    def randn(*shape, scale=1):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale

    q, k, v = [randn(1, total_tokens, heads, dim, scale=0.01) for _ in range(3)]
    gate_dim = dim * 2 if layout == "strided_gates" else dim
    beta_dim = heads * 12 if layout == "strided_gates" else heads
    g = randn(1, total_tokens + 14, heads, gate_dim, scale=0.1)[..., :dim]
    beta = randn(1, total_tokens + 14, beta_dim)[..., :heads]
    initial = (torch.randn(4, heads, dim, dim, device="cuda") * 0.01).to(state_dtype)
    slots = tensor([1] if prefix_len else [1, 2, 0][: len(lengths)])
    batch = SimpleNamespace(
        extend_seq_lens_cpu=lengths,
        mamba_track_seqlens_cpu=[1025] if prefix_len else lengths,
        extend_prefix_lens_cpu=[prefix_len] * len(lengths),
        mamba_prefill_track_mask_cpu=[n > 0 for n in lengths],
    )
    host_plan = build_prefill_track_plan(
        batch.mamba_prefill_track_mask_cpu,
        batch.mamba_track_seqlens_cpu,
        lengths,
        batch.extend_prefix_lens_cpu,
        64,
        mamba2=False,
    )
    metadata = SimpleNamespace(track_ssm_h_batch_src=tensor(host_plan.unaligned_rows))
    build_flashinfer_kda_checkpoint_plan(batch, metadata, "cuda", 64)
    assert metadata.state_checkpoint_cu_starts.tolist() == (
        [0, 9] if prefix_len else [0, 2, 4, 4][: len(lengths) + 1]
    )
    assert metadata.num_state_checkpoints == 1
    assert metadata.state_checkpoint_indices.tolist() == (
        [-1] * 7 + [0, -1] if prefix_len else [-1, 0, -1, -1]
    )
    fi, triton = FlashInferKDAPrefillKernel(), TritonKDAKernel()
    gates = dict(
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.zeros(heads, dim, device="cuda"),
        lower_bound=-5.0,
        beta_is_raw=True,
    )

    def run(kernel, *, track=False, prefix_tokens=None):
        state = initial.clone()
        if layout == "padded_slots" and kernel is fi:
            state = torch.empty_strided(
                initial.shape,
                (heads * dim * dim + 16, dim * dim, dim, 1),
                device="cuda",
                dtype=state_dtype,
            )
            state.copy_(initial)
        run_lengths = lengths if prefix_tokens is None else [prefix_tokens]
        offsets = tensor([0, *accumulate(run_lengths)])
        inputs = [q.clone(), k.clone(), v.clone(), g, beta]
        if prefix_tokens is not None:
            inputs = [x[:, :prefix_tokens] for x in inputs]
        snapshots = torch.full(
            (len(lengths), heads, dim, dim),
            torch.nan,
            device="cuda",
        )
        kwargs = dict(gates, extend_seq_lens_cpu=run_lengths)
        if track:
            kwargs.update(
                return_intermediate_states=True,
                track_state=snapshots,
                track_chunk_idx=tensor(host_plan.chunk_indices),
            )
        if kernel is fi and track:
            kwargs.update(vars(metadata))
        with patch.object(fi, "_recurrent_kda", wraps=fi._recurrent_kda) as recurrent:
            output = kernel.extend(
                *inputs,
                ssm_states=state,
                cache_indices=slots[: len(run_lengths)],
                query_start_loc=offsets,
                **kwargs,
            )
            if kernel is fi and track:
                assert recurrent.call_args.kwargs["backend"] == "cute-dsl"
        assert torch.isfinite(output[0] if track else output).all()
        assert torch.isfinite(state).all()
        return state, snapshots

    fi_state, fi_track = run(fi, track=True)
    assert torch.isfinite(fi_track[0]).all() and torch.isnan(fi_track[1:]).all()
    if layout == "padded_row":
        torch.testing.assert_close(fi_state[0], initial[0], atol=0, rtol=0)
    if not prefix_len:
        _, ref_track = run(triton, track=True)
        assert (
            torch.linalg.vector_norm(fi_track[0] - ref_track[0])
            / (torch.linalg.vector_norm(ref_track[0]) + 1e-12)
            < 5e-2
        )

    boundary = 512 if prefix_len else 128
    # Compare checkpoint and prefix state from the same arithmetic pipeline.
    with patch.object(fi, "_has_small_bh", False):
        prefix_state, _ = run(fi, prefix_tokens=boundary)
    if prefix_len:
        assert (
            torch.linalg.vector_norm(fi_track[0] - prefix_state[1])
            / (torch.linalg.vector_norm(prefix_state[1]) + 1e-12)
            < 5e-2
        )
        triton_prefix, _ = run(triton, prefix_tokens=boundary)
        torch.testing.assert_close(fi_track[0], triton_prefix[1], atol=1e-5, rtol=5e-2)
    else:
        torch.testing.assert_close(fi_track[0], prefix_state[1].float(), atol=0, rtol=0)


@pytest.mark.parametrize("case", ["small_bh", "fp32", "large_bh", "padded_row"])
def test_kda_prefill_pooled_continuation(case):
    """Dense small-BH states must scatter back to the right pool slots.

    Continuing a second packed extend also detects dropped initial states;
    unsupported pool dtypes, large logical batches, and padded rows retain the
    indexed backend even when no intermediate checkpoints are requested.
    """
    if case == "small_bh" and find_spec("flashinfer.kda_prefill_cute_small_bh") is None:
        pytest.skip("The installed FlashInfer predates its small-BH KDA backend")
    torch.manual_seed(19)
    heads, dim = 2, 128
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    lengths = [65, 129]
    if case == "large_bh":
        lengths = [5] * (sm_count // (2 * heads) + 1)
    elif case == "padded_row":
        lengths.append(0)
    batch_size = len(lengths)
    # The physical state pool is deliberately too large for small-BH; dispatch
    # must count packed logical sequences rather than pool capacity or q.shape[0].
    num_slots = max(batch_size + 3, sm_count // (2 * heads) + 3)
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    initial = (torch.randn(num_slots, heads, dim, dim, device="cuda") * 0.1).to(dtype)
    state = torch.empty_strided(
        initial.shape,
        (heads * dim * dim + 16, dim * dim, dim, 1),
        device="cuda",
        dtype=dtype,
    )
    state.copy_(initial)
    reference_state = initial.clone()
    slots = torch.arange(batch_size, 0, -1, device="cuda", dtype=torch.int32)
    untouched = torch.ones(num_slots, device="cuda", dtype=torch.bool)
    untouched[slots[: sum(n > 0 for n in lengths)].long()] = False
    state_ptr = state.data_ptr()
    fi, triton = FlashInferKDAPrefillKernel(), TritonKDAKernel()
    gates = dict(
        A_log=torch.randn(heads, device="cuda") * 0.1,
        dt_bias=torch.randn(heads, dim, device="cuda") * 0.1,
        lower_bound=-5.0,
        beta_is_raw=True,
    )
    expected_backend = "small-bh" if case == "small_bh" else "cute-dsl"

    for step in range(2):
        run_lengths = lengths if step == 0 else [min(n, 67) for n in lengths]
        total_tokens = sum(run_lengths)
        offsets = torch.tensor(
            [0, *accumulate(run_lengths)], device="cuda", dtype=torch.int32
        )
        inputs = [
            torch.randn(
                1, total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16
            )
            * 0.5
            for _ in range(3)
        ]
        # Retaining gates make continuation depend on the previous final state.
        inputs.extend(
            [
                torch.randn(
                    1, total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16
                )
                * 0.3
                - 3.0,
                torch.randn(
                    1, total_tokens, heads, device="cuda", dtype=torch.bfloat16
                ),
            ]
        )
        kwargs = dict(
            gates,
            cache_indices=slots,
            query_start_loc=offsets,
            extend_seq_lens_cpu=run_lengths,
        )
        reference = triton.extend(
            *(x.clone() for x in inputs), ssm_states=reference_state, **kwargs
        )
        tracking = {}
        if step == 1:
            # Tracking may be enabled with no unaligned boundary to snapshot.
            # That must not disable small-BH or modify any tracking destination.
            snapshots = torch.full_like(
                initial[:batch_size], torch.nan, dtype=torch.float32
            )
            starts = list(accumulate((n // 64 for n in run_lengths), initial=0))
            tracking = dict(
                return_intermediate_states=True,
                track_state=snapshots,
                track_ssm_h_batch_src=slots[:0],
                num_state_checkpoints=0,
                state_checkpoint_cu_starts=torch.tensor(
                    starts, device="cuda", dtype=torch.int64
                ),
                state_checkpoint_indices=torch.full(
                    (starts[-1],), -1, device="cuda", dtype=torch.int32
                ),
                state_checkpoint_every_n_tokens=64,
            )
        with patch.object(fi, "_recurrent_kda", wraps=fi._recurrent_kda) as recurrent:
            actual = fi.extend(
                *(x.clone() for x in inputs),
                ssm_states=state,
                **kwargs,
                **tracking,
            )
            assert recurrent.call_args.kwargs["backend"] == expected_backend
        if step == 1:
            actual, intermediate = actual
            assert intermediate is None
            assert torch.isnan(snapshots).all()
        torch.testing.assert_close(actual, reference, atol=5e-4, rtol=3e-2)
        torch.testing.assert_close(state, reference_state, atol=5e-3, rtol=3e-2)
        assert state.data_ptr() == state_ptr
        torch.testing.assert_close(state[untouched], initial[untouched], atol=0, rtol=0)


@pytest.mark.parametrize(
    "extend_lens", [(130, 128), (1,)], ids=["tracked_prefill", "single_token"]
)
def test_kda_backend_prefill_dispatch_and_tracked_state(extend_lens):
    """Raw beta works in FlashInfer and the single-token Triton fallback."""
    with get_parallel().override(attn_dcp_rank=0, attn_dcp_size=1):
        case = KDAAttentionCase(
            name="flashinfer_kda_tracked_extend",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(0,) * len(extend_lens),
            extend_lens=extend_lens,
        )
        fixture = build_kda_attention_fixture(
            CustomTestCase(),
            case,
            head_k_dim=128,
            head_v_dim=128,
            max_context_len=256,
            runner_batch_size=6,
        )
        batch = fixture.forward_batch
        tracked = [n >= 64 for n in extend_lens]
        batch.mamba_track_mask = torch.tensor(tracked, device="cuda")
        batch.mamba_track_indices = torch.arange(
            4, 4 + len(extend_lens), device="cuda", dtype=torch.int32
        )
        batch.mamba_track_seqlens = torch.tensor(
            extend_lens, device="cuda", dtype=torch.int32
        )
        batch.mamba_prefill_track_mask_cpu = tracked
        batch.mamba_track_seqlens_cpu = list(extend_lens)
        fixture.actual_module.attn.lower_bound = -5.0
        cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
        initial_conv, initial_ssm = cache.conv[0].clone(), cache.temporal.clone()

        triton_output = run_kda_fixture_eager(fixture)
        triton_state = cache.temporal.clone()
        triton_conv = cache.conv[0].clone()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        dispatcher = flashinfer_dispatcher()
        backend = fixture.backend.linear_attn_backend
        backend.kernel_dispatcher = dispatcher
        flashinfer_output = run_kda_fixture_eager(fixture)

        torch.testing.assert_close(
            flashinfer_output.float(), triton_output.float(), atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            cache.temporal.float(), triton_state.float(), atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(cache.conv[0], triton_conv, atol=0, rtol=0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
