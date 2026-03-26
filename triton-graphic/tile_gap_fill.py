"""
tile_gap_fill.py – Triton kernels for filling missing TBDR tile regions.

Mirrors graphic-lib GPUGlitchDetector::checkForTileGaps().
Apple's tile-based deferred renderer (TBDR) can fail to flush 32×32 pixel
tiles under heavy load on M4/M5 / A18 GPUs, leaving black rectangles.
This module detects those tiles and fills them with bilinear-interpolated
data from surrounding tiles.
"""
import triton  # type: ignore
import torch  # type: ignore
import triton.language as tl  # type: ignore

TILE_SIZE = 32  # Apple GPU tile size


# ---------------------------------------------------------------------------
# Kernel – detect black (all-zero) tiles
# Each program handles one tile.  Output: 1-element flag per tile.
# ---------------------------------------------------------------------------
@triton.jit
def detect_empty_tile_kernel(
    img_ptr,          # BGRA8 or RGBA float, row-major, 4 components
    flags_ptr,        # output: 1 float per tile (1.0 = empty)
    img_width,
    img_height,
    tile_size,
    tiles_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    tid = tl.program_id(axis=0)
    tx = tid % tiles_per_row
    ty = tid // tiles_per_row

    tile_x = tx * tile_size
    tile_y = ty * tile_size

    total = 0.0
    for dy in range(tile_size):
        row = tile_y + dy
        if row >= img_height:
            break
        for dx in range(tile_size):
            col = tile_x + dx
            if col >= img_width:
                break
            base = (row * img_width + col) * 4
            r = tl.load(img_ptr + base + 0)
            g = tl.load(img_ptr + base + 1)
            b = tl.load(img_ptr + base + 2)
            a = tl.load(img_ptr + base + 3)
            total += r + g + b + a

    is_empty = 1.0 if total == 0.0 else 0.0
    tl.store(flags_ptr + tid, is_empty)


# ---------------------------------------------------------------------------
# Kernel – fill a single empty tile by averaging the 4 surrounding tiles
# ---------------------------------------------------------------------------
@triton.jit
def fill_tile_kernel(
    img_ptr,
    flags_ptr,
    img_width,
    img_height,
    tile_size,
    tiles_per_row,
    tiles_per_col,
    BLOCK_SIZE: tl.constexpr,
):
    tid = tl.program_id(axis=0)
    tx = tid % tiles_per_row
    ty = tid // tiles_per_row

    flag = tl.load(flags_ptr + tid)
    if flag < 0.5:
        return  # tile is fine

    tile_x = tx * tile_size
    tile_y = ty * tile_size

    # Collect average colour from up-to-4 neighbour tiles' centre pixel
    sum_r = 0.0; sum_g = 0.0; sum_b = 0.0; sum_a = 0.0
    count = 0.0
    half = tile_size // 2

    # Left
    if tx > 0:
        cx = (tx - 1) * tile_size + half
        cy = ty * tile_size + half
        if cx < img_width and cy < img_height:
            base = (cy * img_width + cx) * 4
            sum_r += tl.load(img_ptr + base + 0)
            sum_g += tl.load(img_ptr + base + 1)
            sum_b += tl.load(img_ptr + base + 2)
            sum_a += tl.load(img_ptr + base + 3)
            count += 1.0
    # Right
    if tx + 1 < tiles_per_row:
        cx = (tx + 1) * tile_size + half
        cy = ty * tile_size + half
        if cx < img_width and cy < img_height:
            base = (cy * img_width + cx) * 4
            sum_r += tl.load(img_ptr + base + 0)
            sum_g += tl.load(img_ptr + base + 1)
            sum_b += tl.load(img_ptr + base + 2)
            sum_a += tl.load(img_ptr + base + 3)
            count += 1.0
    # Up
    if ty > 0:
        cx = tx * tile_size + half
        cy = (ty - 1) * tile_size + half
        if cx < img_width and cy < img_height:
            base = (cy * img_width + cx) * 4
            sum_r += tl.load(img_ptr + base + 0)
            sum_g += tl.load(img_ptr + base + 1)
            sum_b += tl.load(img_ptr + base + 2)
            sum_a += tl.load(img_ptr + base + 3)
            count += 1.0
    # Down
    if ty + 1 < tiles_per_col:
        cx = tx * tile_size + half
        cy = (ty + 1) * tile_size + half
        if cx < img_width and cy < img_height:
            base = (cy * img_width + cx) * 4
            sum_r += tl.load(img_ptr + base + 0)
            sum_g += tl.load(img_ptr + base + 1)
            sum_b += tl.load(img_ptr + base + 2)
            sum_a += tl.load(img_ptr + base + 3)
            count += 1.0

    if count > 0.0:
        inv = 1.0 / count
        fill_r = sum_r * inv
        fill_g = sum_g * inv
        fill_b = sum_b * inv
        fill_a = sum_a * inv
    else:
        fill_r = 0.0; fill_g = 0.0; fill_b = 0.0; fill_a = 0.0

    # Paint entire tile with the averaged colour
    for dy in range(tile_size):
        row = tile_y + dy
        if row >= img_height:
            break
        for dx in range(tile_size):
            col = tile_x + dx
            if col >= img_width:
                break
            base = (row * img_width + col) * 4
            tl.store(img_ptr + base + 0, fill_r)
            tl.store(img_ptr + base + 1, fill_g)
            tl.store(img_ptr + base + 2, fill_b)
            tl.store(img_ptr + base + 3, fill_a)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def detect_tile_gaps(
    image: torch.Tensor, width: int, height: int, tile_size: int = TILE_SIZE
) -> torch.Tensor:
    """Return a 1-D float tensor with one flag per tile (1.0 = empty)."""
    assert image.is_cuda
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    n_tiles = tiles_x * tiles_y

    flags = torch.zeros(n_tiles, device=image.device, dtype=image.dtype)
    flat = image.contiguous().view(-1)

    detect_empty_tile_kernel[(n_tiles,)](
        flat, flags, width, height, tile_size, tiles_x, BLOCK_SIZE=1,
    )
    return flags


def fill_tile_gaps(
    image: torch.Tensor, width: int, height: int, tile_size: int = TILE_SIZE
) -> torch.Tensor:
    """Detect empty TBDR tiles and fill them with neighbour averages (in‑place)."""
    assert image.is_cuda
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    n_tiles = tiles_x * tiles_y

    flags = detect_tile_gaps(image, width, height, tile_size)

    flat = image.contiguous().view(-1)
    fill_tile_kernel[(n_tiles,)](
        flat, flags, width, height, tile_size, tiles_x, tiles_y, BLOCK_SIZE=1,
    )
    return image


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    W, H = 128, 128
    img = torch.rand(H, W, 4, device="cuda")
    # Simulate a missing tile at (32,32)-(63,63)
    img[32:64, 32:64, :] = 0.0

    gaps_before = detect_tile_gaps(img, W, H).sum().item()
    fill_tile_gaps(img, W, H)
    gaps_after = detect_tile_gaps(img, W, H).sum().item()
    print(f"Tile gaps: {int(gaps_before)} -> {int(gaps_after)}")
