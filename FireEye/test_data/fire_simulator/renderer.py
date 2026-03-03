"""
Fire Simulation Renderer — Pure Python PNG Output

Renders fire simulation grids to PNG images using ONLY the Python standard
library (struct + zlib for PNG encoding). No matplotlib, no PIL, no numpy.

Features:
- Color-coded cell states with smooth gradients
- Fire glow effects (radial light from burning cells)
- Smoke overlay with transparency blending
- HUD overlay with simulation stats
- Scalable pixel size (each cell = NxN pixels)
- Optional wind direction indicator
"""

import struct
import zlib
import math
import os
from typing import List, Tuple, Optional, Dict

from simulator import CellState, FuelType, FUEL_PROPERTIES, FireSimulator, Cell


# ── PNG Writer (pure stdlib) ──────────────────────────────────────────

def _make_png(width: int, height: int, pixels: List[List[Tuple[int, int, int]]]) -> bytes:
    """
    Create a PNG file from a 2D array of (R, G, B) tuples.
    pixels[row][col] = (r, g, b) where each is 0-255.
    """
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + c + struct.pack('>I', crc)

    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b'IHDR', ihdr_data)

    # IDAT chunk — raw pixel data with filter bytes
    raw_data = bytearray()
    for row in pixels:
        raw_data.append(0)  # Filter: None
        for r, g, b in row:
            raw_data.append(min(255, max(0, r)))
            raw_data.append(min(255, max(0, g)))
            raw_data.append(min(255, max(0, b)))

    compressed = zlib.compress(bytes(raw_data), 9)
    idat = chunk(b'IDAT', compressed)

    # IEND chunk
    iend = chunk(b'IEND', b'')

    return sig + ihdr + idat + iend


# ── Color Palettes ────────────────────────────────────────────────────

# Base colors for cell states (R, G, B)
COLORS = {
    CellState.EMPTY:   (30, 30, 25),       # Dark ground
    CellState.FUEL:    (40, 80, 30),        # Green (vegetation) — overridden by fuel type
    CellState.BURNING: (255, 120, 0),       # Orange fire
    CellState.BURNED:  (50, 40, 35),        # Dark ash/char
    CellState.SMOKE:   (120, 120, 120),     # Gray smoke
    CellState.WALL:    (130, 130, 135),      # Concrete gray
    CellState.EMBER:   (180, 60, 20),       # Glowing red
}

# Fuel-type specific colors (when cell is FUEL state)
FUEL_COLORS = {
    FuelType.NONE:     (30, 30, 25),
    FuelType.GRASS:    (55, 110, 35),       # Green
    FuelType.WOOD:     (100, 70, 40),       # Brown
    FuelType.FABRIC:   (80, 60, 90),        # Purple-ish
    FuelType.CONCRETE: (130, 130, 135),     # Gray
    FuelType.BRUSH:    (45, 90, 30),        # Dark green
}

# Fire color gradient based on burn_timer (young fire = yellow, old = deep red)
FIRE_GRADIENT = [
    (255, 255, 100),    # 0: bright yellow
    (255, 220, 50),     # 1: yellow-orange
    (255, 180, 20),     # 2: orange
    (255, 130, 0),      # 3: deep orange
    (230, 80, 0),       # 4: red-orange
    (200, 50, 10),      # 5: red
    (170, 30, 10),      # 6: dark red
    (140, 20, 5),       # 7: very dark red
]


def _lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int],
                t: float) -> Tuple[int, int, int]:
    """Linear interpolation between two colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _blend(base: Tuple[int, int, int], overlay: Tuple[int, int, int],
           alpha: float) -> Tuple[int, int, int]:
    """Blend overlay onto base with given alpha (0=base, 1=overlay)."""
    alpha = max(0.0, min(1.0, alpha))
    return (
        int(base[0] * (1 - alpha) + overlay[0] * alpha),
        int(base[1] * (1 - alpha) + overlay[1] * alpha),
        int(base[2] * (1 - alpha) + overlay[2] * alpha),
    )


def _brighten(color: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    """Brighten a color by adding amount (0-1 scale mapped to 0-255)."""
    add = int(amount * 255)
    return (
        min(255, color[0] + add),
        min(255, color[1] + add),
        min(255, color[2] + add),
    )


# ── 3x5 Bitmap Font for HUD ─────────────────────────────────────────

FONT_3X5 = {
    '0': ['111', '101', '101', '101', '111'],
    '1': ['010', '110', '010', '010', '111'],
    '2': ['111', '001', '111', '100', '111'],
    '3': ['111', '001', '111', '001', '111'],
    '4': ['101', '101', '111', '001', '001'],
    '5': ['111', '100', '111', '001', '111'],
    '6': ['111', '100', '111', '101', '111'],
    '7': ['111', '001', '001', '001', '001'],
    '8': ['111', '101', '111', '101', '111'],
    '9': ['111', '101', '111', '001', '111'],
    'A': ['010', '101', '111', '101', '101'],
    'B': ['110', '101', '110', '101', '110'],
    'C': ['111', '100', '100', '100', '111'],
    'D': ['110', '101', '101', '101', '110'],
    'E': ['111', '100', '110', '100', '111'],
    'F': ['111', '100', '110', '100', '100'],
    'G': ['111', '100', '101', '101', '111'],
    'H': ['101', '101', '111', '101', '101'],
    'I': ['111', '010', '010', '010', '111'],
    'K': ['101', '110', '100', '110', '101'],
    'L': ['100', '100', '100', '100', '111'],
    'M': ['101', '111', '111', '101', '101'],
    'N': ['101', '111', '111', '111', '101'],
    'O': ['111', '101', '101', '101', '111'],
    'P': ['111', '101', '111', '100', '100'],
    'R': ['110', '101', '110', '101', '101'],
    'S': ['111', '100', '111', '001', '111'],
    'T': ['111', '010', '010', '010', '010'],
    'U': ['101', '101', '101', '101', '111'],
    'W': ['101', '101', '111', '111', '101'],
    'X': ['101', '101', '010', '101', '101'],
    'Y': ['101', '101', '010', '010', '010'],
    ':': ['000', '010', '000', '010', '000'],
    ' ': ['000', '000', '000', '000', '000'],
    '.': ['000', '000', '000', '000', '010'],
    '/': ['001', '001', '010', '100', '100'],
    '-': ['000', '000', '111', '000', '000'],
    '%': ['101', '001', '010', '100', '101'],
    '#': ['010', '111', '010', '111', '010'],
    '=': ['000', '111', '000', '111', '000'],
    '+': ['000', '010', '111', '010', '000'],
}


def _draw_text(pixels, x: int, y: int, text: str,
               color: Tuple[int, int, int] = (255, 255, 255),
               scale: int = 1):
    """Draw text onto pixel buffer using 3x5 bitmap font."""
    height = len(pixels)
    width = len(pixels[0]) if pixels else 0
    cx = x
    for ch in text.upper():
        glyph = FONT_3X5.get(ch, FONT_3X5.get(' '))
        if glyph is None:
            cx += 4 * scale
            continue
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == '1':
                    for sy in range(scale):
                        for sx in range(scale):
                            px = cx + gx * scale + sx
                            py = y + gy * scale + sy
                            if 0 <= py < height and 0 <= px < width:
                                pixels[py][px] = color
        cx += (len(glyph[0]) + 1) * scale


def _draw_rect_outline(pixels, x1, y1, x2, y2,
                       color: Tuple[int, int, int]):
    """Draw a rectangle outline on the pixel buffer."""
    height = len(pixels)
    width = len(pixels[0]) if pixels else 0
    for x in range(max(0, x1), min(width, x2 + 1)):
        if 0 <= y1 < height:
            pixels[y1][x] = color
        if 0 <= y2 < height:
            pixels[y2][x] = color
    for y in range(max(0, y1), min(height, y2 + 1)):
        if 0 <= x1 < width:
            pixels[y][x1] = color
        if 0 <= x2 < width:
            pixels[y][x2] = color


def _draw_filled_rect(pixels, x1, y1, x2, y2,
                      color: Tuple[int, int, int]):
    """Draw a filled rectangle on the pixel buffer."""
    height = len(pixels)
    width = len(pixels[0]) if pixels else 0
    for y in range(max(0, y1), min(height, y2 + 1)):
        for x in range(max(0, x1), min(width, x2 + 1)):
            pixels[y][x] = color


# ── Main Renderer ─────────────────────────────────────────────────────

class FireRenderer:
    """Renders fire simulation state to PNG images."""

    def __init__(self, cell_size: int = 6, show_hud: bool = True,
                 show_glow: bool = True, show_wind: bool = True):
        """
        cell_size: pixels per grid cell
        show_hud: draw text overlay with stats
        show_glow: render fire glow effects
        show_wind: show wind direction arrow
        """
        self.cell_size = cell_size
        self.show_hud = show_hud
        self.show_glow = show_glow
        self.show_wind = show_wind

    def render(self, sim: FireSimulator, label: str = "") -> bytes:
        """Render the current simulation state to PNG bytes."""
        cs = self.cell_size
        img_w = sim.width * cs
        img_h = sim.height * cs

        # HUD bar at the bottom
        hud_h = 30 if self.show_hud else 0
        total_h = img_h + hud_h

        # Initialize pixel buffer (black)
        pixels = [[(0, 0, 0) for _ in range(img_w)] for _ in range(total_h)]

        # ── Pass 1: Base cell colors ──
        for r in range(sim.height):
            for c in range(sim.width):
                cell = sim.grid[r][c]
                color = self._cell_color(cell)

                # Fill cell pixels
                y0 = r * cs
                x0 = c * cs
                for dy in range(cs):
                    for dx in range(cs):
                        py = y0 + dy
                        px = x0 + dx
                        if 0 <= py < img_h and 0 <= px < img_w:
                            pixels[py][px] = color

        # ── Pass 2: Fire glow effects ──
        if self.show_glow:
            glow_radius = int(self.cell_size * 2.5)
            for r in range(sim.height):
                for c in range(sim.width):
                    cell = sim.grid[r][c]
                    if cell.state == CellState.BURNING:
                        intensity = 0.25 + 0.15 * min(1.0, cell.burn_timer / 3.0)
                        cy = r * cs + cs // 2
                        cx = c * cs + cs // 2
                        for dy in range(-glow_radius, glow_radius + 1):
                            for dx in range(-glow_radius, glow_radius + 1):
                                py = cy + dy
                                px = cx + dx
                                if 0 <= py < img_h and 0 <= px < img_w:
                                    dist = math.sqrt(dy * dy + dx * dx)
                                    if dist <= glow_radius:
                                        falloff = 1.0 - (dist / glow_radius)
                                        glow_amount = intensity * falloff * falloff
                                        glow_color = (255, 150, 30)
                                        pixels[py][px] = _blend(
                                            pixels[py][px], glow_color, glow_amount
                                        )
                    elif cell.state == CellState.EMBER:
                        # Subtle red glow for embers
                        ember_radius = int(self.cell_size * 1.2)
                        cy = r * cs + cs // 2
                        cx = c * cs + cs // 2
                        for dy in range(-ember_radius, ember_radius + 1):
                            for dx in range(-ember_radius, ember_radius + 1):
                                py = cy + dy
                                px = cx + dx
                                if 0 <= py < img_h and 0 <= px < img_w:
                                    dist = math.sqrt(dy * dy + dx * dx)
                                    if dist <= ember_radius:
                                        falloff = 1.0 - (dist / ember_radius)
                                        glow_amount = 0.1 * cell.temperature * falloff
                                        pixels[py][px] = _blend(
                                            pixels[py][px], (200, 50, 10), glow_amount
                                        )

        # ── Pass 3: Smoke overlay ──
        for r in range(sim.height):
            for c in range(sim.width):
                cell = sim.grid[r][c]
                if cell.smoke_density > 0.02:
                    y0 = r * cs
                    x0 = c * cs
                    smoke_alpha = cell.smoke_density * 0.5
                    smoke_color = (160, 160, 165)
                    for dy in range(cs):
                        for dx in range(cs):
                            py = y0 + dy
                            px = x0 + dx
                            if 0 <= py < img_h and 0 <= px < img_w:
                                pixels[py][px] = _blend(
                                    pixels[py][px], smoke_color, smoke_alpha
                                )

        # ── Pass 4: Temperature heat shimmer on fuel cells ──
        for r in range(sim.height):
            for c in range(sim.width):
                cell = sim.grid[r][c]
                if cell.state == CellState.FUEL and cell.temperature > 0.1:
                    y0 = r * cs
                    x0 = c * cs
                    heat_alpha = cell.temperature * 0.3
                    for dy in range(cs):
                        for dx in range(cs):
                            py = y0 + dy
                            px = x0 + dx
                            if 0 <= py < img_h and 0 <= px < img_w:
                                pixels[py][px] = _blend(
                                    pixels[py][px], (255, 100, 0), heat_alpha
                                )

        # ── Pass 5: Grid lines (subtle) ──
        if cs >= 4:
            grid_color = (20, 20, 18)
            for r in range(sim.height + 1):
                py = r * cs
                if 0 <= py < img_h:
                    for px in range(img_w):
                        pixels[py][px] = _blend(pixels[py][px], grid_color, 0.3)
            for c_idx in range(sim.width + 1):
                px = c_idx * cs
                if 0 <= px < img_w:
                    for py in range(img_h):
                        pixels[py][px] = _blend(pixels[py][px], grid_color, 0.3)

        # ── Pass 6: Wind direction indicator ──
        if self.show_wind and sim.wind.speed > 0.01:
            self._draw_wind_arrow(pixels, img_w, img_h, sim.wind)

        # ── Pass 7: HUD overlay ──
        if self.show_hud:
            stats = sim.get_stats()
            self._draw_hud(pixels, img_w, img_h, hud_h, stats, label, sim.wind)

        return _make_png(img_w, total_h, pixels)

    def _cell_color(self, cell: Cell) -> Tuple[int, int, int]:
        """Determine the base color for a cell."""
        if cell.state == CellState.FUEL:
            return FUEL_COLORS.get(cell.fuel_type, FUEL_COLORS[FuelType.NONE])

        elif cell.state == CellState.BURNING:
            # Use fire gradient based on burn timer
            idx = min(int(cell.burn_timer), len(FIRE_GRADIENT) - 1)
            if idx < len(FIRE_GRADIENT) - 1:
                frac = cell.burn_timer - int(cell.burn_timer)
                return _lerp_color(FIRE_GRADIENT[idx], FIRE_GRADIENT[idx + 1], frac)
            return FIRE_GRADIENT[idx]

        elif cell.state == CellState.EMBER:
            # Embers glow based on temperature
            base = (80, 30, 10)
            glow = (220, 80, 15)
            return _lerp_color(base, glow, cell.temperature)

        elif cell.state == CellState.BURNED:
            # Ash — slightly varies
            base = COLORS[CellState.BURNED]
            return base

        elif cell.state == CellState.WALL:
            return COLORS[CellState.WALL]

        else:
            return COLORS[CellState.EMPTY]

    def _draw_wind_arrow(self, pixels, img_w, img_h, wind):
        """Draw a wind direction arrow in the top-right corner."""
        # Arrow center
        cx = img_w - 25
        cy = 25
        length = 10 + int(wind.speed * 10)

        # Arrow tip
        rad = math.radians(wind.direction)
        tip_x = cx + int(math.sin(rad) * length)
        tip_y = cy + int(-math.cos(rad) * length)
        tail_x = cx - int(math.sin(rad) * length * 0.5)
        tail_y = cy + int(math.cos(rad) * length * 0.5)

        # Draw line (Bresenham-ish)
        self._draw_line(pixels, tail_x, tail_y, tip_x, tip_y, (200, 200, 255))

        # Arrowhead
        head_len = 5
        for angle_off in [-150, 150]:
            a = math.radians(wind.direction + angle_off)
            hx = tip_x + int(math.sin(a) * head_len)
            hy = tip_y + int(-math.cos(a) * head_len)
            self._draw_line(pixels, tip_x, tip_y, hx, hy, (200, 200, 255))

        # Label "W"
        _draw_text(pixels, cx - 3, cy - length - 10, "W", (200, 200, 255), scale=1)

    def _draw_line(self, pixels, x0, y0, x1, y1, color):
        """Draw a line using Bresenham's algorithm."""
        height = len(pixels)
        width = len(pixels[0]) if pixels else 0
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= y0 < height and 0 <= x0 < width:
                pixels[y0][x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _draw_hud(self, pixels, img_w, img_h, hud_h, stats, label, wind):
        """Draw the heads-up display bar at the bottom."""
        # Background bar
        _draw_filled_rect(pixels, 0, img_h, img_w - 1, img_h + hud_h - 1, (15, 15, 20))
        # Top border
        for x in range(img_w):
            pixels[img_h][x] = (80, 80, 80)

        y_text = img_h + 4
        scale = 2

        # Frame number
        frame_str = f"T:{stats['timestep']}"
        _draw_text(pixels, 4, y_text, frame_str, (200, 200, 200), scale=scale)

        # Burning count
        burn_str = f"FIRE:{stats['burning']}"
        _draw_text(pixels, 80, y_text, burn_str, (255, 130, 0), scale=scale)

        # Fuel remaining
        fuel_str = f"FUEL:{stats['fuel']}"
        _draw_text(pixels, 200, y_text, fuel_str, (55, 160, 55), scale=scale)

        # Burned
        burned_str = f"ASH:{stats['burned']}"
        _draw_text(pixels, 320, y_text, burned_str, (140, 120, 100), scale=scale)

        # Label / scene name
        if label:
            _draw_text(pixels, img_w - len(label) * 4 * scale - 8, y_text,
                       label, (180, 180, 200), scale=scale)


def render_frame(sim: FireSimulator, output_path: str,
                 cell_size: int = 6, label: str = "",
                 show_hud: bool = True, show_glow: bool = True):
    """Convenience function: render one frame to a PNG file."""
    renderer = FireRenderer(cell_size=cell_size, show_hud=show_hud,
                            show_glow=show_glow)
    png_data = renderer.render(sim, label=label)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(png_data)
    return output_path


def render_sequence(sim: FireSimulator, output_dir: str, num_frames: int = 30,
                    cell_size: int = 6, label: str = "",
                    show_hud: bool = True, show_glow: bool = True,
                    steps_per_frame: int = 1) -> List[str]:
    """
    Run simulation for num_frames steps, rendering each to a PNG.
    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    renderer = FireRenderer(cell_size=cell_size, show_hud=show_hud,
                            show_glow=show_glow)
    paths = []

    for i in range(num_frames):
        # Render current state
        frame_label = f"{label} F{i}" if label else f"F{i}"
        png_data = renderer.render(sim, label=frame_label)
        path = os.path.join(output_dir, f"frame_{i:04d}.png")
        with open(path, 'wb') as f:
            f.write(png_data)
        paths.append(path)

        # Advance simulation
        for _ in range(steps_per_frame):
            sim.step()

    return paths
