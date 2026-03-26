import triton  # type: ignore
import torch  # type: ignore
import triton.language as tl  # type: ignore


# ---------------------------------------------------------------------------
# Kernel 1 – Per-channel clamp (works on flat RGBA interleaved data)
# ---------------------------------------------------------------------------
@triton.jit
def fix_rgba_channel_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Clamp every RGBA component to [0.0, 1.0] and flush NaN/Inf to 0."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Replace NaN / Inf with 0 first
    is_bad = (x != x) | (x == float("inf")) | (x == float("-inf"))
    x = tl.where(is_bad, 0.0, x)

    # Clamp to valid colour range
    x = tl.where(x > 1.0, 1.0, x)
    x = tl.where(x < 0.0, 0.0, x)

    tl.store(output_ptr + offsets, x, mask=mask)


# Legacy alias kept for backward-compat
fix_blue_channel_kernel = fix_rgba_channel_kernel


# ---------------------------------------------------------------------------
# Kernel 2 – Half-precision denorm floor
# Mirrors the C++ ShaderPatchManager "half-precision denorm flush" fix.
# On Apple M4/M5 GPUs, half-float denormals are flushed to zero
# inconsistently across SIMD lanes, causing banding artefacts.
# This kernel clamps |x| < min_normal to ±min_normal so that
# downstream rendering sees consistent non-zero values.
# ---------------------------------------------------------------------------
@triton.jit
def half_denorm_floor_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_normal,            # fp16 min normal ≈ 6.1035e-5
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    abs_x = tl.where(x < 0.0, -x, x)
    sign = tl.where(x < 0.0, -1.0, 1.0)

    # If the magnitude is non-zero but smaller than min_normal, bump it up
    is_denorm = (abs_x > 0.0) & (abs_x < min_normal)
    x = tl.where(is_denorm, sign * min_normal, x)

    tl.store(output_ptr + offsets, x, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def fix_graphics_buffer(input_tensor: torch.Tensor) -> torch.Tensor:
    """Fix iPhone Metal GPU buffer graphics – full RGBA clamp + NaN flush."""
    assert input_tensor.is_cuda, "Tensor must be on a GPU device"

    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    fix_rgba_channel_kernel[grid](
        input_tensor, output, n_elements, BLOCK_SIZE=1024,
    )
    return output


def fix_half_denorms(input_tensor: torch.Tensor) -> torch.Tensor:
    """Flush half-precision denormals to ±min_normal (Apple M4/M5 fix)."""
    assert input_tensor.is_cuda, "Tensor must be on a GPU device"

    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    FP16_MIN_NORMAL = 6.103515625e-5
    half_denorm_floor_kernel[grid](
        input_tensor, output, n_elements, FP16_MIN_NORMAL, BLOCK_SIZE=1024,
    )
    return output


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_data = torch.randn(1024 * 1024, device="cuda") * 2 - 1
    corrected = fix_graphics_buffer(test_data)
    print(f"RGBA fix: {corrected.min().item():.4f} .. {corrected.max().item():.4f}")

    tiny = torch.tensor([1e-6, -3e-6, 0.5, 0.0], device="cuda")
    fixed = fix_half_denorms(tiny)
    print(f"Denorm fix: {fixed.tolist()}")