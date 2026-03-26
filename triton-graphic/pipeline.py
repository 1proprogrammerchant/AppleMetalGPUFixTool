"""
pipeline.py – High-level GPU fix pipeline for iPhone graphics on Triton.

Ties together every Triton kernel module into a single class that can be
called once per frame (or on-demand) to detect and repair all known
Apple GPU 2026 glitch categories.
"""
import torch  # type: ignore

from tritongraphicset import fix_graphics_buffer, fix_half_denorms
from texture_repair import scrub_nan_inf, repair_texture
from depth_fix import fix_depth_buffer
from tile_gap_fill import fill_tile_gaps, detect_tile_gaps
from flicker_suppress import suppress_flicker, compute_flicker_mask
from vertex_sanitizer import sanitize_vertices, count_outlier_vertices


class TritonGPUFixPipeline:
    """
    Drop-in GPU fix pipeline for iPhone / Apple Silicon graphics.

    Usage::

        pipe = TritonGPUFixPipeline(device="cuda")
        pipe.capture_reference(good_frame)

        # Each frame:
        frame = pipe.fix_frame(raw_frame, width, height)
        verts = pipe.fix_vertices(raw_verts)
        depth = pipe.fix_depth(raw_depth, width, height)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._reference_frame: torch.Tensor | None = None
        self._stats = {
            "frames_fixed": 0,
            "textures_repaired": 0,
            "tiles_filled": 0,
            "flickers_suppressed": 0,
            "vertices_sanitized": 0,
            "depth_fixes": 0,
        }

    # ------------------------------------------------------------------
    # Reference frame management (for flicker suppression)
    # ------------------------------------------------------------------
    def capture_reference(self, frame: torch.Tensor) -> None:
        """Store a known-good frame as the flicker-suppression reference."""
        self._reference_frame = frame.clone()

    # ------------------------------------------------------------------
    # Full frame fix (runs all applicable passes)
    # ------------------------------------------------------------------
    def fix_frame(
        self,
        frame: torch.Tensor,
        width: int,
        height: int,
        *,
        fix_tiles: bool = True,
        fix_flicker: bool = True,
        fix_denorms: bool = True,
        flicker_threshold: float = 0.03,
        flicker_blend: float = 0.8,
    ) -> torch.Tensor:
        """
        Apply the complete fix pipeline to a single RGBA frame tensor.

        Parameters
        ----------
        frame : torch.Tensor
            H×W×4 float32 RGBA image on GPU.
        width, height : int
            Frame dimensions.
        fix_tiles : bool
            Detect and fill missing TBDR tiles.
        fix_flicker : bool
            Suppress single-frame flicker against reference.
        fix_denorms : bool
            Clamp half-precision denormals.
        """
        # 1. NaN / Inf scrub
        flat = frame.contiguous().view(-1)
        scrub_nan_inf(flat)

        # 2. RGBA value clamp
        flat = fix_graphics_buffer(flat)
        frame = flat.view(height, width, 4)

        # 3. Texture corruption – neighbour-blend any remaining bad pixels
        repair_texture(frame, width, height)
        self._stats["textures_repaired"] += 1

        # 4. Half-precision denorm floor
        if fix_denorms:
            flat = frame.contiguous().view(-1)
            flat = fix_half_denorms(flat)
            frame = flat.view(height, width, 4)

        # 5. TBDR tile gap fill
        if fix_tiles:
            gaps = detect_tile_gaps(frame, width, height)
            n_gaps = int(gaps.sum().item())
            if n_gaps > 0:
                fill_tile_gaps(frame, width, height)
                self._stats["tiles_filled"] += n_gaps

        # 6. Flicker suppression
        if fix_flicker and self._reference_frame is not None:
            mask = compute_flicker_mask(
                self._reference_frame, frame, flicker_threshold
            )
            if mask.sum().item() > 0:
                frame = suppress_flicker(
                    self._reference_frame, frame, flicker_threshold, flicker_blend
                )
                self._stats["flickers_suppressed"] += 1

        # Update reference for next frame
        self._reference_frame = frame.clone()
        self._stats["frames_fixed"] += 1
        return frame

    # ------------------------------------------------------------------
    # Vertex fix
    # ------------------------------------------------------------------
    def fix_vertices(
        self, positions: torch.Tensor, bound: float = 1e6
    ) -> torch.Tensor:
        """Clamp exploded vertices and flush NaN/Inf."""
        sanitize_vertices(positions, bound)
        self._stats["vertices_sanitized"] += 1
        return positions

    # ------------------------------------------------------------------
    # Depth buffer fix
    # ------------------------------------------------------------------
    def fix_depth(
        self,
        depth: torch.Tensor,
        width: int,
        height: int,
        threshold: float = 0.95,
    ) -> torch.Tensor:
        """Smooth depth-buffer z-fighting artefacts."""
        result = fix_depth_buffer(depth, width, height, threshold)
        self._stats["depth_fixes"] += 1
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def report(self) -> dict:
        """Return cumulative fix statistics."""
        return dict(self._stats)

    def print_report(self) -> None:
        print("\n===== Triton GPU Fix Pipeline Report =====")
        for k, v in self._stats.items():
            print(f"  {k:.<30s} {v}")
        print("==========================================\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipe = TritonGPUFixPipeline()

    W, H = 128, 128

    # Simulate a bad frame
    frame = torch.rand(H, W, 4, device="cuda")
    frame[10, 20] = float("nan")       # texture corruption
    frame[32:64, 32:64, :] = 0.0       # missing tile
    frame[50, 50, 2] = 5.0             # blue overflow

    fixed = pipe.fix_frame(frame, W, H)
    print(f"Frame range: [{fixed.min().item():.4f}, {fixed.max().item():.4f}]")
    print(f"NaN remaining: {torch.isnan(fixed).any().item()}")

    # Vertex fix
    verts = torch.randn(5000, 3, device="cuda")
    verts[0] = float("inf")
    pipe.fix_vertices(verts)
    print(f"Outliers after fix: {count_outlier_vertices(verts)}")

    # Depth fix
    depth = torch.rand(H, W, device="cuda")
    depth[64, 64] = 0.01
    depth[64, 65] = 0.99
    pipe.fix_depth(depth, W, H, threshold=0.5)

    pipe.print_report()
