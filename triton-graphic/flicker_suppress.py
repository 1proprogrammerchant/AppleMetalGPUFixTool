"""
flicker_suppress.py – Triton kernels for frame-to-frame flicker suppression.

Mirrors graphic-lib FrameBufferValidator.
Compares the current frame against a reference and, where the per-pixel
difference exceeds a threshold, blends toward the reference to suppress
the visible flicker caused by Apple GPU TBDR timing glitches.
"""
import triton  # type: ignore
import torch  # type: ignore
import triton.language as tl  # type: ignore


# ---------------------------------------------------------------------------
# Kernel – per-pixel difference mask (RGBA float, row-major)
# Output: 1.0 where any channel differs by > threshold, else 0.0
# ---------------------------------------------------------------------------
@triton.jit
def frame_diff_kernel(
    ref_ptr,
    cur_ptr,
    mask_ptr,
    n_pixels,       # total pixels (W*H)
    threshold,      # per-component float threshold
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    base = offsets * 4  # 4 components per pixel

    r_ref = tl.load(ref_ptr + base + 0, mask=mask, other=0.0)
    g_ref = tl.load(ref_ptr + base + 1, mask=mask, other=0.0)
    b_ref = tl.load(ref_ptr + base + 2, mask=mask, other=0.0)
    a_ref = tl.load(ref_ptr + base + 3, mask=mask, other=0.0)

    r_cur = tl.load(cur_ptr + base + 0, mask=mask, other=0.0)
    g_cur = tl.load(cur_ptr + base + 1, mask=mask, other=0.0)
    b_cur = tl.load(cur_ptr + base + 2, mask=mask, other=0.0)
    a_cur = tl.load(cur_ptr + base + 3, mask=mask, other=0.0)

    dr = tl.where(r_cur > r_ref, r_cur - r_ref, r_ref - r_cur)
    dg = tl.where(g_cur > g_ref, g_cur - g_ref, g_ref - g_cur)
    db = tl.where(b_cur > b_ref, b_cur - b_ref, b_ref - b_cur)
    da = tl.where(a_cur > a_ref, a_cur - a_ref, a_ref - a_cur)

    max_d = tl.maximum(tl.maximum(dr, dg), tl.maximum(db, da))
    is_flicker = max_d > threshold
    tl.store(mask_ptr + offsets, tl.where(is_flicker, 1.0, 0.0), mask=mask)


# ---------------------------------------------------------------------------
# Kernel – blend flickering pixels toward reference
# out = lerp(current, reference, blend_factor) where mask == 1
# ---------------------------------------------------------------------------
@triton.jit
def flicker_blend_kernel(
    ref_ptr,
    cur_ptr,
    out_ptr,
    mask_ptr,
    n_pixels,
    blend_factor,   # 0.0 = keep current, 1.0 = fully replace with ref
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_pixels

    m = tl.load(mask_ptr + offsets, mask=valid, other=0.0)
    base = offsets * 4

    for c in range(4):
        ref_v = tl.load(ref_ptr + base + c, mask=valid, other=0.0)
        cur_v = tl.load(cur_ptr + base + c, mask=valid, other=0.0)

        blended = cur_v + (ref_v - cur_v) * blend_factor
        result = tl.where(m > 0.5, blended, cur_v)
        tl.store(out_ptr + base + c, result, mask=valid)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def compute_flicker_mask(
    reference: torch.Tensor,
    current: torch.Tensor,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Return a per-pixel mask (H*W) flagging flickering pixels."""
    assert reference.is_cuda and current.is_cuda
    assert reference.shape == current.shape
    n_pixels = reference.numel() // 4
    mask = torch.zeros(n_pixels, device=reference.device, dtype=reference.dtype)

    grid = lambda meta: (triton.cdiv(n_pixels, meta["BLOCK_SIZE"]),)
    frame_diff_kernel[grid](
        reference.contiguous().view(-1),
        current.contiguous().view(-1),
        mask,
        n_pixels,
        threshold,
        BLOCK_SIZE=1024,
    )
    return mask


def suppress_flicker(
    reference: torch.Tensor,
    current: torch.Tensor,
    threshold: float = 0.03,
    blend: float = 0.8,
) -> torch.Tensor:
    """
    Suppress frame-to-frame flicker.

    Pixels that differ from |reference| by more than |threshold| are blended
    back toward the reference by |blend| factor, eliminating single-frame
    glitch artefacts from the Apple TBDR pipeline.
    """
    assert reference.is_cuda and current.is_cuda
    n_pixels = reference.numel() // 4
    mask = compute_flicker_mask(reference, current, threshold)

    out = torch.empty_like(current)
    grid = lambda meta: (triton.cdiv(n_pixels, meta["BLOCK_SIZE"]),)
    flicker_blend_kernel[grid](
        reference.contiguous().view(-1),
        current.contiguous().view(-1),
        out.view(-1),
        mask,
        n_pixels,
        blend,
        BLOCK_SIZE=1024,
    )
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 64, 64
    ref = torch.rand(H, W, 4, device="cuda")
    cur = ref.clone()
    # Simulate flicker on a patch
    cur[20:30, 20:30, :] += 0.5

    fixed = suppress_flicker(ref, cur, threshold=0.03, blend=0.9)
    diff = (fixed - ref).abs().max().item()
    print(f"Max diff after suppression: {diff:.4f}")
