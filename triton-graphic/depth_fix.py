"""
depth_fix.py – Triton kernels for depth-buffer artefact correction.

Mirrors graphic-lib GPUGlitchDetector::checkDepthBuffer().
Detects extreme depth discontinuities (z-fighting) and smooths them
with a bilateral-style horizontal filter on the GPU.
"""
import triton  # type: ignore
import torch  # type: ignore
import triton.language as tl  # type: ignore


# ---------------------------------------------------------------------------
# Kernel – flag depth discontinuities
# Output: mask tensor, 1.0 where |d[x]-d[x-1]| > threshold, else 0.0
# ---------------------------------------------------------------------------
@triton.jit
def depth_discontinuity_kernel(
    depth_ptr,
    mask_ptr,
    width,
    height,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= height:
        return

    row_off = row * width
    cols = tl.arange(0, BLOCK_SIZE)

    # Process row in chunks of BLOCK_SIZE
    for start in range(0, width, BLOCK_SIZE):
        idx = start + cols
        valid = idx < width

        d = tl.load(depth_ptr + row_off + idx, mask=valid, other=0.0)

        # Shift left by 1 – for col 0 use d itself (no delta)
        prev_idx = idx - 1
        prev_valid = valid & (prev_idx >= 0)
        d_prev = tl.load(depth_ptr + row_off + prev_idx, mask=prev_valid, other=d)

        delta = tl.where(d > d_prev, d - d_prev, d_prev - d)
        is_disc = (delta > threshold) & valid
        tl.store(mask_ptr + row_off + idx, tl.where(is_disc, 1.0, 0.0), mask=valid)


# ---------------------------------------------------------------------------
# Kernel – smooth depth at flagged locations (horizontal 3-tap average)
# ---------------------------------------------------------------------------
@triton.jit
def depth_smooth_kernel(
    depth_ptr,
    mask_ptr,
    out_ptr,
    width,
    height,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= height:
        return

    row_off = row * width
    cols = tl.arange(0, BLOCK_SIZE)

    for start in range(0, width, BLOCK_SIZE):
        idx = start + cols
        valid = idx < width

        d = tl.load(depth_ptr + row_off + idx, mask=valid, other=0.0)
        m = tl.load(mask_ptr + row_off + idx, mask=valid, other=0.0)

        # Only smooth flagged pixels
        needs_fix = m > 0.5

        left_idx = idx - 1
        right_idx = idx + 1
        has_left = left_idx >= 0
        has_right = right_idx < width

        d_left = tl.load(
            depth_ptr + row_off + left_idx, mask=valid & has_left, other=d
        )
        d_right = tl.load(
            depth_ptr + row_off + right_idx, mask=valid & has_right, other=d
        )

        count = 1.0 + tl.where(has_left, 1.0, 0.0) + tl.where(has_right, 1.0, 0.0)
        avg = (d_left + d + d_right) / count

        result = tl.where(needs_fix, avg, d)
        tl.store(out_ptr + row_off + idx, result, mask=valid)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def detect_depth_artifacts(
    depth: torch.Tensor, width: int, height: int, threshold: float = 0.95
) -> torch.Tensor:
    """Return a float mask (same shape as depth) flagging discontinuities."""
    assert depth.is_cuda
    mask = torch.zeros_like(depth)
    flat_depth = depth.contiguous().view(-1)
    flat_mask = mask.view(-1)

    BLOCK = min(1024, width)
    depth_discontinuity_kernel[(height,)](
        flat_depth, flat_mask, width, height, threshold, BLOCK_SIZE=BLOCK,
    )
    return mask


def fix_depth_buffer(
    depth: torch.Tensor, width: int, height: int, threshold: float = 0.95
) -> torch.Tensor:
    """Detect and smooth depth discontinuities in a W×H depth tensor."""
    assert depth.is_cuda
    flat_depth = depth.contiguous().view(-1)

    mask = torch.zeros(width * height, device=depth.device, dtype=depth.dtype)
    out = torch.empty_like(flat_depth)

    BLOCK = min(1024, width)

    # Pass 1: detect
    depth_discontinuity_kernel[(height,)](
        flat_depth, mask, width, height, threshold, BLOCK_SIZE=BLOCK,
    )
    # Pass 2: smooth
    depth_smooth_kernel[(height,)](
        flat_depth, mask, out, width, height, BLOCK_SIZE=BLOCK,
    )
    return out.view_as(depth)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    W, H = 128, 128
    depth = torch.rand(H, W, device="cuda")
    # Inject harsh discontinuity
    depth[64, 64] = 0.01
    depth[64, 65] = 0.99

    fixed = fix_depth_buffer(depth, W, H, threshold=0.5)
    print(f"Depth fix applied. Max delta after fix: "
          f"{(fixed[1:] - fixed[:-1]).abs().max().item():.4f}")
