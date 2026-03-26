"""
texture_repair.py – Triton kernels for repairing corrupted GPU textures.

Mirrors graphic-lib GPUGlitchDetector::checkTextureIntegrity().
Scans RGBA float textures for NaN/Inf and replaces them with
neighbour-averaged values so the frame is visually coherent.
"""
import triton  # type: ignore
import torch  # type: ignore
import triton.language as tl  # type: ignore


# ---------------------------------------------------------------------------
# Kernel – replace NaN / Inf pixels with zero (fast single-pass)
# ---------------------------------------------------------------------------
@triton.jit
def nan_inf_scrub_kernel(
    ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place: replace NaN and ±Inf with 0.0."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(ptr + offsets, mask=mask, other=0.0)
    is_bad = (x != x) | (x == float("inf")) | (x == float("-inf"))
    x = tl.where(is_bad, 0.0, x)
    tl.store(ptr + offsets, x, mask=mask)


# ---------------------------------------------------------------------------
# Kernel – horizontal neighbour blend for corrupted pixels
# Works on a flat RGBA buffer laid out row-major:
#   [R G B A  R G B A  ...] with 4 floats per pixel, width pixels per row.
# A pixel is "corrupted" if any component is NaN or Inf.
# Corrupted pixels are replaced with the average of their left and right
# neighbours (or a single neighbour at the row edges).
# ---------------------------------------------------------------------------
@triton.jit
def neighbour_blend_kernel(
    ptr,
    width,          # pixels per row
    height,         # number of rows
    BLOCK_SIZE: tl.constexpr,
):
    """Per-row horizontal neighbour blend for bad pixels (in-place)."""
    row = tl.program_id(axis=0)
    if row >= height:
        return

    row_base = row * width * 4  # 4 components per pixel

    for px in range(0, width):
        base = row_base + px * 4
        r = tl.load(ptr + base + 0)
        g = tl.load(ptr + base + 1)
        b = tl.load(ptr + base + 2)
        a = tl.load(ptr + base + 3)

        is_bad = (
            (r != r) | (r == float("inf")) | (r == float("-inf"))
            | (g != g) | (g == float("inf")) | (g == float("-inf"))
            | (b != b) | (b == float("inf")) | (b == float("-inf"))
            | (a != a) | (a == float("inf")) | (a == float("-inf"))
        )

        if is_bad:
            lr = 0.0; lg = 0.0; lb = 0.0; la = 0.0
            rr = 0.0; rg = 0.0; rb = 0.0; ra = 0.0
            count = 0.0

            if px > 0:
                lbase = row_base + (px - 1) * 4
                lr = tl.load(ptr + lbase + 0)
                lg = tl.load(ptr + lbase + 1)
                lb = tl.load(ptr + lbase + 2)
                la = tl.load(ptr + lbase + 3)
                if lr == lr:  # not NaN
                    count += 1.0
                else:
                    lr = 0.0; lg = 0.0; lb = 0.0; la = 0.0

            if px < width - 1:
                rbase = row_base + (px + 1) * 4
                rr = tl.load(ptr + rbase + 0)
                rg = tl.load(ptr + rbase + 1)
                rb = tl.load(ptr + rbase + 2)
                ra = tl.load(ptr + rbase + 3)
                if rr == rr:
                    count += 1.0
                else:
                    rr = 0.0; rg = 0.0; rb = 0.0; ra = 0.0

            if count > 0.0:
                inv = 1.0 / count
                tl.store(ptr + base + 0, (lr + rr) * inv)
                tl.store(ptr + base + 1, (lg + rg) * inv)
                tl.store(ptr + base + 2, (lb + rb) * inv)
                tl.store(ptr + base + 3, (la + ra) * inv)
            else:
                tl.store(ptr + base + 0, 0.0)
                tl.store(ptr + base + 1, 0.0)
                tl.store(ptr + base + 2, 0.0)
                tl.store(ptr + base + 3, 0.0)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def scrub_nan_inf(tensor: torch.Tensor) -> torch.Tensor:
    """In-place NaN/Inf → 0 scrub on a flat GPU tensor."""
    assert tensor.is_cuda
    n = tensor.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    nan_inf_scrub_kernel[grid](tensor, n, BLOCK_SIZE=1024)
    return tensor


def repair_texture(texture: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Repair a GPU texture tensor (H×W×4, float32, row-major RGBA).

    Corrupted (NaN/Inf) pixels are replaced with neighbour averages.
    Operates in-place and returns the same tensor.
    """
    assert texture.is_cuda
    assert texture.numel() == width * height * 4, "Expected W*H*4 elements"

    flat = texture.contiguous().view(-1)
    neighbour_blend_kernel[(height,)](flat, width, height, BLOCK_SIZE=1)
    return texture


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    W, H = 64, 64
    tex = torch.rand(H, W, 4, device="cuda")
    # Inject some corruption
    tex[10, 20] = float("nan")
    tex[30, 40] = float("inf")

    repair_texture(tex, W, H)
    has_nan = torch.isnan(tex).any().item()
    has_inf = torch.isinf(tex).any().item()
    print(f"After repair: NaN={has_nan}, Inf={has_inf}")
