"""Parametric synthetic fire scene generator for FireEye Future Agent testing.

Each scene is saved in two versions:
  *_clean.png      — visual elements only, NO text  → fed to the AI agent
  *_annotated.png  — same image + all text labels    → human reference

Scenes:
  A — Small contained flame, isolated         (expected future risk: LOW)
  B — Small flame + wood stack 80 px away    (expected future risk: MEDIUM)
  C — Large roaring flame + embers, open     (expected future risk: HIGH)
  D — Large flame + embers + materials near
      + wind                                  (expected future risk: CRITICAL)

Usage:
    python3 generate_test_scenes.py [--output-dir test_data/synthetic]
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Canvas / layout constants
# ─────────────────────────────────────────────────────────────────────────────
W, H = 640, 480
FLOOR_Y       = 340
WALL_COLOUR   = (228, 224, 215)
FLOOR_COLOUR  = (175, 172, 165)
GRID_COLOUR   = (155, 152, 145)


# ─────────────────────────────────────────────────────────────────────────────
# Font helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    try:
        return ImageFont.load_default(size=size)      # type: ignore[call-arg]
    except TypeError:
        return ImageFont.load_default()


FONT_BIG   = _get_font(18)
FONT_MED   = _get_font(13)
FONT_SMALL = _get_font(10)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except AttributeError:
        return len(text) * 6


# ─────────────────────────────────────────────────────────────────────────────
# Background
# ─────────────────────────────────────────────────────────────────────────────

def draw_background(img: Image.Image, floor_y: int = FLOOR_Y) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0),       (W, floor_y)], fill=WALL_COLOUR)
    draw.rectangle([(0, floor_y), (W, H)],       fill=FLOOR_COLOUR)
    for x in range(0, W + 1, 64):
        draw.line([(x, floor_y), (x, H)], fill=GRID_COLOUR, width=1)
    for y in range(floor_y, H + 1, 48):
        draw.line([(0, y), (W, y)], fill=GRID_COLOUR, width=1)
    draw.line([(0, floor_y), (W, floor_y)], fill=(110, 105, 98), width=3)


# ─────────────────────────────────────────────────────────────────────────────
# Flame drawing
# ─────────────────────────────────────────────────────────────────────────────

def _flame_polygon(
    cx: int, base_y: int,
    height: int, width: int,
    lean: int = 0, jitter: int = 0,
    tip_frac: float = 1.0,
    rng: random.Random | None = None,
    n_side: int = 8,
) -> list[tuple[int, int]]:
    if rng is None:
        rng = random.Random(0)
    eff_h = int(height * tip_frac)
    pts: list[tuple[int, int]] = []
    pts.append((cx - width // 2, base_y))
    for i in range(1, n_side):
        t = i / n_side
        w = (width / 2) * ((1 - t) ** 0.65)
        jag = rng.randint(-jitter, jitter)
        pts.append((int(cx - w + lean * t + jag), base_y - int(eff_h * t)))
    tip_jag = rng.randint(-jitter // 2, jitter // 2) if jitter else 0
    pts.append((cx + lean + tip_jag, base_y - eff_h))
    for i in range(n_side - 1, 0, -1):
        t = i / n_side
        w = (width / 2) * ((1 - t) ** 0.65)
        jag = rng.randint(-jitter, jitter)
        pts.append((int(cx + w + lean * t + jag), base_y - int(eff_h * t)))
    pts.append((cx + width // 2, base_y))
    return pts


def draw_flame(
    img: Image.Image,
    cx: int, base_y: int,
    height: int, width: int,
    intensity: str = "calm",
    lean: int = 0,
    seed: int = 42,
) -> None:
    rng     = random.Random(seed)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)

    if intensity == "calm":
        layers = [
            (1.00, 0, (255, 110,  15, 180)),
            (0.85, 0, (255, 185,  45, 200)),
            (0.65, 0, (255, 240, 100, 220)),
        ]
        n_side = 8
    else:
        layers = [
            (1.00, width // 5, (200,  30,   0, 155)),
            (0.97, width // 4, (255,  70,   5, 165)),
            (0.85, width // 6, (255, 150,  20, 185)),
            (0.70, width // 8, (255, 215,  55, 205)),
            (0.50, 2,          (255, 255, 200, 230)),
        ]
        n_side = 11

    for tip_frac, jitter, color in layers:
        pts = _flame_polygon(
            cx, base_y, height, width,
            lean=lean, jitter=jitter, tip_frac=tip_frac,
            rng=rng, n_side=n_side,
        )
        odraw.polygon(pts, fill=color)

    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


# ─────────────────────────────────────────────────────────────────────────────
# Embers
# ─────────────────────────────────────────────────────────────────────────────

def draw_embers(
    img: Image.Image,
    cx: int, base_y: int,
    flame_height: int,
    count: int, spread_radius: int,
    lean: int = 0,
    seed: int = 99,
) -> None:
    if count == 0:
        return
    rng     = random.Random(seed)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)
    tip_x   = cx + lean
    tip_y   = base_y - flame_height
    bias    = math.atan2(lean, flame_height) * 0.8

    for _ in range(count):
        angle = rng.uniform(-math.pi * 0.65, math.pi * 0.65) + bias
        dist  = rng.uniform(15, spread_radius)
        ex    = tip_x + int(dist * math.sin(angle))
        ey    = tip_y - int(dist * abs(math.cos(angle)) * 0.7)
        r     = rng.choices([2, 3, 4, 5], weights=[50, 30, 15, 5])[0]
        heat  = max(0.0, 1 - dist / spread_radius)
        gc    = int(80 + 160 * heat)
        alpha = int(180 + 50 * heat)
        odraw.ellipse([(ex - r, ey - r), (ex + r, ey + r)],
                      fill=(255, gc, int(10 * heat), alpha))
        if r >= 4:
            trail = rng.randint(6, 16)
            tx = ex - int(trail * math.sin(angle) * 0.6)
            ty = ey + int(trail * 0.4)
            odraw.line([(ex, ey), (tx, ty)], fill=(255, 140, 0, 100), width=1)

    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


# ─────────────────────────────────────────────────────────────────────────────
# Flammable material blocks  (shapes only — no text)
# ─────────────────────────────────────────────────────────────────────────────

_MAT_SPEC: dict[str, tuple] = {
    "wood":      ((160, 100,  45), (120,  72,  28)),
    "cardboard": ((210, 175,  95), (165, 130,  55)),
    "fabric":    ((145,  80, 155), (105,  50, 115)),
    "gas_can":   ((210,  40,  40), (160,  20,  20)),
}

_MAT_LABEL: dict[str, tuple[str, tuple]] = {
    "wood":      ("WOOD",      (255, 255, 255)),
    "cardboard": ("CARD\nBOX", ( 60,  40,  10)),
    "fabric":    ("FABRIC",    (255, 255, 255)),
    "gas_can":   ("GAS",       (255, 255, 255)),
}


def draw_material(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    material_type: str,
) -> None:
    """Draw material block shape and texture — NO text label."""
    light, dark = _MAT_SPEC.get(material_type, ((150, 150, 150), (110, 110, 110)))
    draw.rectangle([(x, y), (x + w, y + h)], fill=light, outline=dark, width=2)
    if material_type == "wood":
        for gy in range(y + 6, y + h - 2, 8):
            draw.line([(x + 3, gy), (x + w - 3, gy)], fill=dark, width=1)
    elif material_type == "cardboard":
        for gx in range(x + 5, x + w - 2, 7):
            draw.line([(gx, y + 2), (gx, y + h - 2)], fill=dark, width=1)
    elif material_type == "fabric":
        for gx in range(x + 4, x + w - 2, 10):
            for gy in range(y + 4, y + h - 2, 10):
                draw.ellipse([(gx, gy), (gx + 4, gy + 4)], outline=dark, width=1)
    elif material_type == "gas_can":
        nw = w // 4
        draw.rectangle(
            [(x + w // 2 - nw // 2, y - 8), (x + w // 2 + nw // 2, y)],
            fill=dark,
        )
        draw.line([(x + 4, y + h // 2), (x + w - 4, y + h // 2)],
                  fill=(255, 220, 0), width=3)


def label_material(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    material_type: str,
    font: ImageFont.ImageFont | None = None,
) -> None:
    """Draw the centred text label over a material block."""
    if font is None:
        font = FONT_SMALL
    label, txt_col = _MAT_LABEL.get(
        material_type, (material_type.upper(), (20, 20, 20))
    )
    lines   = label.split("\n")
    line_h  = 12
    total_h = len(lines) * line_h
    start_y = y + (h - total_h) // 2
    for i, line in enumerate(lines):
        tw = _text_width(draw, line, font)
        draw.text((x + (w - tw) // 2, start_y + i * line_h),
                  line, fill=txt_col, font=font)


# ─────────────────────────────────────────────────────────────────────────────
# Wind arrow  (arrow shape only — no text)
# ─────────────────────────────────────────────────────────────────────────────

def draw_wind_arrow(
    draw: ImageDraw.ImageDraw,
    x: int, y: int,
    direction: str = "right",
) -> None:
    """Draw wind arrow indicator — NO label."""
    col = (80, 130, 210)
    arrow_len = 60
    if direction == "right":
        x1, y1, x2, y2 = x, y, x + arrow_len, y
    else:
        x1, y1, x2, y2 = x + arrow_len, y, x, y
    draw.line([(x1, y1), (x2, y2)], fill=col, width=3)
    sign = 1 if x2 > x1 else -1
    draw.polygon(
        [(x2, y2), (x2 - sign * 12, y2 - 7), (x2 - sign * 12, y2 + 7)],
        fill=col,
    )
    for i in range(1, 4):
        fx = x1 + (arrow_len // 4) * i
        draw.line([(fx, y1 - 6), (fx, y1 + 6)], fill=col, width=2)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation-only helpers  (never called on clean images)
# ─────────────────────────────────────────────────────────────────────────────

def ann_scene_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    pad      = 6
    line_gap = 20
    total_h  = 28 + (line_gap if subtitle else 0) + pad
    banner_w = max(len(title), len(subtitle)) * 10 + pad * 2
    draw.rectangle([(4, 4), (banner_w + 4, total_h + 4)], fill=(15, 15, 15))
    draw.text((4 + pad, 4 + pad), title,    fill=(255, 255,  80), font=FONT_BIG)
    if subtitle:
        draw.text((4 + pad, 4 + pad + line_gap), subtitle, fill=(200, 200, 200), font=FONT_MED)


_RISK_COL = {
    "LOW":      ( 60, 160,  60),
    "MEDIUM":   (210, 140,  10),
    "HIGH":     (210,  60,  10),
    "CRITICAL": (180,  10,  10),
}


def ann_risk_legend(draw: ImageDraw.ImageDraw, expected_future_risk: str) -> None:
    col  = _RISK_COL.get(expected_future_risk.upper(), (150, 150, 150))
    text = f"Expected Future Risk: {expected_future_risk.upper()}"
    pad  = 6
    tw   = _text_width(draw, text, FONT_MED)
    x    = W - tw - pad * 2 - 6
    y    = H - 22 - pad * 2 - 6
    draw.rectangle([(x, y), (W - 6, H - 6)], fill=(15, 15, 15))
    draw.text((x + pad, y + pad), text, fill=col, font=FONT_MED)


def ann_distance(
    draw: ImageDraw.ImageDraw,
    x1: int, x2: int, y: int,
    label: str,
) -> None:
    col = (200, 50, 50)
    draw.line([(x1, y), (x2, y)], fill=col, width=1)
    draw.line([(x1, y - 5), (x1, y + 5)], fill=col, width=2)
    draw.line([(x2, y - 5), (x2, y + 5)], fill=col, width=2)
    mid = (x1 + x2) // 2
    tw  = _text_width(draw, label, FONT_SMALL)
    draw.text((mid - tw // 2, y + 6), label, fill=col, font=FONT_SMALL)


def ann_wind_label(
    draw: ImageDraw.ImageDraw,
    x: int, y: int,
    label: str = "WIND",
) -> None:
    draw.text((x, y - 20), label, fill=(80, 130, 210), font=FONT_MED)


# ─────────────────────────────────────────────────────────────────────────────
# Burn mark
# ─────────────────────────────────────────────────────────────────────────────

def draw_burn_mark(draw: ImageDraw.ImageDraw, cx: int, base_y: int, rx: int, ry: int) -> None:
    draw.ellipse(
        [(cx - rx, base_y - ry // 2), (cx + rx, base_y + ry)],
        fill=(50, 38, 25),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ──  SCENE A  ──  Small contained flame, isolated  ──────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def make_scene_A(out_dir: Path) -> None:
    img = Image.new("RGB", (W, H))
    draw_background(img)

    cx = W // 2

    # Candle body (visual only)
    draw = ImageDraw.Draw(img)
    candle_x, candle_w, candle_h = cx - 10, 20, 55
    draw.rectangle(
        [(candle_x, FLOOR_Y), (candle_x + candle_w, FLOOR_Y + candle_h)],
        fill=(245, 245, 215), outline=(180, 175, 130), width=2,
    )
    draw.ellipse([(candle_x + 5, FLOOR_Y + 50), (candle_x + 15, FLOOR_Y + 60)],
                 fill=(230, 230, 200))

    draw_flame(img, cx, FLOOR_Y, height=52, width=28, intensity="calm", seed=1)

    # ── CLEAN save ────────────────────────────────────────────────────────
    img.save(out_dir / "scene_A_small_contained_clean.png")
    print(f"  Saved: {out_dir}/scene_A_small_contained_clean.png")

    # ── Annotated copy ────────────────────────────────────────────────────
    ann = img.copy()
    adraw = ImageDraw.Draw(ann)
    adraw.text((candle_x - 8, FLOOR_Y + candle_h + 4), "CANDLE",
               fill=(80, 70, 55), font=FONT_SMALL)
    adraw.text((60, FLOOR_Y + 30),    "Empty floor", fill=(130, 120, 110), font=FONT_SMALL)
    adraw.text((W - 130, FLOOR_Y + 30), "Empty floor", fill=(130, 120, 110), font=FONT_SMALL)
    adraw.text((cx - 50, FLOOR_Y + 90),
               "No flammable materials nearby\nNo wind  |  No particles",
               fill=(100, 90, 80), font=FONT_SMALL)
    ann_scene_header(adraw, "SCENE A", "Small Contained Flame — Isolated")
    ann_risk_legend(adraw, "LOW")
    ann.save(out_dir / "scene_A_small_contained_annotated.png")
    print(f"  Saved: {out_dir}/scene_A_small_contained_annotated.png")


# ─────────────────────────────────────────────────────────────────────────────
# ──  SCENE B  ──  Small flame + wood stack nearby  ──────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def make_scene_B(out_dir: Path) -> None:
    img = Image.new("RGB", (W, H))
    draw_background(img)

    cx = W // 2 - 90

    draw = ImageDraw.Draw(img)
    candle_x, candle_w, candle_h = cx - 10, 20, 55
    draw.rectangle(
        [(candle_x, FLOOR_Y), (candle_x + candle_w, FLOOR_Y + candle_h)],
        fill=(245, 245, 215), outline=(180, 175, 130), width=2,
    )

    draw_flame(img, cx, FLOOR_Y, height=55, width=30, intensity="calm", seed=2)

    draw = ImageDraw.Draw(img)
    mat_x    = cx + 80
    plank_w, plank_h = 80, 22
    for row in range(3):
        draw_material(draw, mat_x, FLOOR_Y + row * (plank_h - 2), plank_w, plank_h, "wood")

    # Cardboard further away
    card_x = cx + 240
    draw_material(draw, card_x, FLOOR_Y, 60, 50, "cardboard")

    # ── CLEAN save ────────────────────────────────────────────────────────
    img.save(out_dir / "scene_B_small_near_material_clean.png")
    print(f"  Saved: {out_dir}/scene_B_small_near_material_clean.png")

    # ── Annotated copy ────────────────────────────────────────────────────
    ann   = img.copy()
    adraw = ImageDraw.Draw(ann)
    adraw.text((candle_x - 8, FLOOR_Y + candle_h + 4), "CANDLE",
               fill=(80, 70, 55), font=FONT_SMALL)
    for row in range(3):
        label_material(adraw, mat_x, FLOOR_Y + row * (plank_h - 2),
                       plank_w, plank_h, "wood")
    adraw.text((mat_x + 5, FLOOR_Y + 3 * (plank_h - 2) + 4),
               "WOOD PLANKS", fill=(90, 55, 20), font=FONT_SMALL)
    label_material(adraw, card_x, FLOOR_Y, 60, 50, "cardboard")
    adraw.text((card_x + 3, FLOOR_Y + 54), "CARDBOARD", fill=(90, 60, 10), font=FONT_SMALL)
    ann_distance(adraw, cx + 15, mat_x, FLOOR_Y + 80, "~80 px  (≈ 0.5 m)")
    ann_scene_header(adraw, "SCENE B", "Small Flame + Nearby Wood Stack")
    ann_risk_legend(adraw, "MEDIUM")
    ann.save(out_dir / "scene_B_small_near_material_annotated.png")
    print(f"  Saved: {out_dir}/scene_B_small_near_material_annotated.png")


# ─────────────────────────────────────────────────────────────────────────────
# ──  SCENE C  ──  Large roaring flame + embers, open space  ─────────────────
# ─────────────────────────────────────────────────────────────────────────────

def make_scene_C(out_dir: Path) -> None:
    img     = Image.new("RGB", (W, H))
    draw_background(img)

    cx      = W // 2
    flame_h = 185
    flame_w = 95

    draw = ImageDraw.Draw(img)
    draw_burn_mark(draw, cx, FLOOR_Y, rx=55, ry=14)

    draw_flame(img, cx, FLOOR_Y, height=flame_h, width=flame_w,
               intensity="roaring", seed=3)
    draw_embers(img, cx, FLOOR_Y, flame_h, count=40, spread_radius=170, seed=10)

    # Ember landing-zone ellipse (visual, no text)
    draw = ImageDraw.Draw(img)
    lz_r = 165
    draw.ellipse(
        [(cx - lz_r, FLOOR_Y - lz_r // 2), (cx + lz_r, FLOOR_Y + lz_r // 2)],
        outline=(220, 80, 10), width=1,
    )

    # ── CLEAN save ────────────────────────────────────────────────────────
    img.save(out_dir / "scene_C_large_roaring_open_clean.png")
    print(f"  Saved: {out_dir}/scene_C_large_roaring_open_clean.png")

    # ── Annotated copy ────────────────────────────────────────────────────
    ann   = img.copy()
    adraw = ImageDraw.Draw(ann)
    adraw.text((cx + 110, FLOOR_Y - flame_h - 10),
               "EMBER\nPLUME", fill=(255, 140, 40), font=FONT_MED)
    adraw.line([(cx + 108, FLOOR_Y - flame_h - 8), (cx + 80, FLOOR_Y - flame_h + 20)],
               fill=(255, 140, 40), width=2)
    adraw.text((cx + lz_r - 60, FLOOR_Y - lz_r // 2 - 14),
               "ember landing zone", fill=(220, 80, 10), font=FONT_SMALL)
    adraw.text((30, FLOOR_Y + 30),    "Open space", fill=(130, 120, 110), font=FONT_SMALL)
    adraw.text((W - 120, FLOOR_Y + 30), "Open space", fill=(130, 120, 110), font=FONT_SMALL)
    ann_scene_header(adraw, "SCENE C", "Large Roaring Flame + Embers — Open Space")
    ann_risk_legend(adraw, "HIGH")
    ann.save(out_dir / "scene_C_large_roaring_open_annotated.png")
    print(f"  Saved: {out_dir}/scene_C_large_roaring_open_annotated.png")


# ─────────────────────────────────────────────────────────────────────────────
# ──  SCENE D  ──  Critical cascade  ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def make_scene_D(out_dir: Path) -> None:
    img     = Image.new("RGB", (W, H))
    draw_background(img)

    cx      = W // 2 - 20
    flame_h = 215
    flame_w = 120
    lean    = 45

    draw = ImageDraw.Draw(img)
    draw_burn_mark(draw, cx + lean // 2, FLOOR_Y, rx=70, ry=18)

    draw_flame(img, cx, FLOOR_Y, height=flame_h, width=flame_w,
               intensity="roaring", lean=lean, seed=4)
    draw_embers(img, cx, FLOOR_Y, flame_h, count=75, spread_radius=230,
                lean=lean, seed=20)

    draw = ImageDraw.Draw(img)

    # Material shapes (no labels)
    gx = cx - flame_w // 2 - 42
    draw_material(draw, gx, FLOOR_Y - 10, 35, 50, "gas_can")

    wx = cx + flame_w // 2 + 15
    for row in range(3):
        draw_material(draw, wx, FLOOR_Y + row * 20, 80, 19, "wood")

    bx = 30
    draw_material(draw, bx, FLOOR_Y - 5, 75, 65, "cardboard")

    fx = W - 100
    draw_material(draw, fx, FLOOR_Y, 80, 45, "fabric")

    # Wind arrow (no label)
    draw_wind_arrow(draw, W - 140, 60, direction="right")

    # ── CLEAN save ────────────────────────────────────────────────────────
    img.save(out_dir / "scene_D_critical_cascade_clean.png")
    print(f"  Saved: {out_dir}/scene_D_critical_cascade_clean.png")

    # ── Annotated copy ────────────────────────────────────────────────────
    ann   = img.copy()
    adraw = ImageDraw.Draw(ann)

    label_material(adraw, gx, FLOOR_Y - 10, 35, 50, "gas_can")
    adraw.text((gx - 5, FLOOR_Y + 44), "⚠ CLOSE!", fill=(255, 80, 80), font=FONT_SMALL)

    for row in range(3):
        label_material(adraw, wx, FLOOR_Y + row * 20, 80, 19, "wood")
    adraw.text((wx, FLOOR_Y + 62), "WOOD PLANKS", fill=(130, 80, 20), font=FONT_SMALL)

    label_material(adraw, bx, FLOOR_Y - 5, 75, 65, "cardboard")
    adraw.text((bx + 3, FLOOR_Y + 64), "CARDBOARD", fill=(100, 70, 10), font=FONT_SMALL)

    label_material(adraw, fx, FLOOR_Y, 80, 45, "fabric")
    adraw.text((fx + 3, FLOOR_Y + 48), "FABRIC", fill=(90, 40, 100), font=FONT_SMALL)

    ann_wind_label(adraw, W - 140, 60, label="STRONG WIND")
    ann_distance(adraw, gx + 35, cx - flame_w // 2, FLOOR_Y + 100, "~42 px")
    ann_distance(adraw, cx + flame_w // 2, wx,       FLOOR_Y + 100, "~15 px")
    adraw.text((cx + lean + 115, FLOOR_Y - flame_h + 20),
               "EMBER\nSTORM", fill=(255, 100, 20), font=FONT_MED)
    ann_scene_header(adraw, "SCENE D", "Critical: Large Flame + Embers + Materials + Wind")
    ann_risk_legend(adraw, "CRITICAL")
    ann.save(out_dir / "scene_D_critical_cascade_annotated.png")
    print(f"  Saved: {out_dir}/scene_D_critical_cascade_annotated.png")


# ─────────────────────────────────────────────────────────────────────────────
# Scene registry
# ─────────────────────────────────────────────────────────────────────────────

# name_stem → (generator function, expected future risk)
SCENES: dict[str, tuple] = {
    "scene_A_small_contained":     (make_scene_A, "LOW"),
    "scene_B_small_near_material": (make_scene_B, "MEDIUM"),
    "scene_C_large_roaring_open":  (make_scene_C, "HIGH"),
    "scene_D_critical_cascade":    (make_scene_D, "CRITICAL"),
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate parametric synthetic FireEye test scenes"
    )
    parser.add_argument(
        "--output-dir", default="test_data/synthetic",
        help="Directory to save generated images  [default: test_data/synthetic]",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic FireEye test scenes …")
    for name, (fn, _) in SCENES.items():
        print(f"\n  [{name}]")
        fn(out_dir)

    print(f"\nDone — 8 files saved to {out_dir}/  (4 clean + 4 annotated)")
    print()
    print("Key parametric distinctions:")
    print("  flame_height : A=52  B=55  C=185  D=215")
    print("  intensity    : A/B=calm          C/D=roaring")
    print("  ember_count  : A=0   B=0   C=40   D=75")
    print("  materials    : none  wood  none   gas+wood+card+fabric")
    print("  wind lean    : 0     0     0      45 px")
    print()
    print("Clean files (fed to AI agent):")
    for name in SCENES:
        print(f"  {name}_clean.png")
    print()
    print("Annotated files (human reference):")
    for name in SCENES:
        print(f"  {name}_annotated.png")


if __name__ == "__main__":
    main()
