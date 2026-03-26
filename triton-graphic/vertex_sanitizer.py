"""
vertex_sanitizer.py – Triton kernels for vertex buffer sanitisation.

Mirrors graphic-lib GPUGlitchDetector::checkVertexBuffer().
Detects and repairs "vertex explosions" where positions fly to extreme
coordinates or become NaN/Inf due to Apple GPU pipeline bugs.
"""
import triton  # type: ignore
import torch  # type: ignore
import triton.language as tl  # type: ignore

DEFAULT_BOUND = 1e6  # positions beyond this are considered outliers


# ---------------------------------------------------------------------------
# Kernel – clamp vertex positions (x, y, z) that exceed a bounding box
# Vertex layout: tightly packed float3 (stride = 3 floats per vertex)
# ---------------------------------------------------------------------------
@triton.jit
def vertex_clamp_kernel(
    ptr,            # float* – vertex positions, x y z x y z ...
    n_verts,
    bound,          # max absolute coordinate value
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Each "element" is a single float component (3 per vertex)
    n_components = n_verts * 3
    valid = offsets < n_components

    v = tl.load(ptr + offsets, mask=valid, other=0.0)

    # NaN / Inf → 0
    is_bad = (v != v) | (v == float("inf")) | (v == float("-inf"))
    v = tl.where(is_bad, 0.0, v)

    # Clamp to [-bound, +bound]
    v = tl.where(v > bound, bound, v)
    v = tl.where(v < -bound, -bound, v)

    tl.store(ptr + offsets, v, mask=valid)


# ---------------------------------------------------------------------------
# Kernel – count outlier vertices (returns per-block partial sums)
# ---------------------------------------------------------------------------
@triton.jit
def vertex_outlier_count_kernel(
    ptr,
    counts_ptr,     # output: one int per block
    n_verts,
    bound,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    base = pid * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_verts

    # Check x, y, z for each vertex
    x = tl.load(ptr + offsets * 3 + 0, mask=valid, other=0.0)
    y = tl.load(ptr + offsets * 3 + 1, mask=valid, other=0.0)
    z = tl.load(ptr + offsets * 3 + 2, mask=valid, other=0.0)

    def is_outlier(v):
        return (v != v) | (v == float("inf")) | (v == float("-inf")) | (v > bound) | (v < -bound)

    bad = is_outlier(x) | is_outlier(y) | is_outlier(z)
    bad_count = tl.sum(tl.where(bad & valid, 1, 0))
    tl.store(counts_ptr + pid, bad_count)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def count_outlier_vertices(
    positions: torch.Tensor, bound: float = DEFAULT_BOUND
) -> int:
    """Count vertices with any component outside [-bound, +bound] or NaN/Inf."""
    assert positions.is_cuda
    n_verts = positions.numel() // 3
    BLOCK = 1024
    n_blocks = (n_verts + BLOCK - 1) // BLOCK
    counts = torch.zeros(n_blocks, device=positions.device, dtype=torch.int32)

    vertex_outlier_count_kernel[(n_blocks,)](
        positions.contiguous().view(-1), counts, n_verts, bound, BLOCK_SIZE=BLOCK,
    )
    return int(counts.sum().item())


def sanitize_vertices(
    positions: torch.Tensor, bound: float = DEFAULT_BOUND
) -> torch.Tensor:
    """
    In-place clamp vertex positions to [-bound, +bound] and flush NaN/Inf.

    Expects a contiguous float tensor of shape (N, 3) or (N*3,).
    """
    assert positions.is_cuda
    n_components = positions.numel()
    n_verts = n_components // 3
    BLOCK = 1024

    grid = lambda meta: (triton.cdiv(n_components, meta["BLOCK_SIZE"]),)
    vertex_clamp_kernel[grid](
        positions.contiguous().view(-1), n_verts, bound, BLOCK_SIZE=BLOCK,
    )
    return positions


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    verts = torch.randn(10000, 3, device="cuda")
    # Inject some explosions
    verts[42] = float("nan")
    verts[99] = torch.tensor([1e8, -1e8, float("inf")], device="cuda")

    bad_before = count_outlier_vertices(verts)
    sanitize_vertices(verts)
    bad_after = count_outlier_vertices(verts)
    print(f"Outlier vertices: {bad_before} -> {bad_after}")
