#!/usr/bin/env python3
"""
Extended Visual Cryptography Scheme (EVCS) for grayscale images.

Based on Nakajima & Yamaguchi (2002) and Liu & Wu (2011). The script
generates two meaningful shares from two cover images and one secret image.
Stacking the two shares reveals the secret.
"""

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# 1. Image loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str, size: int) -> np.ndarray:
    """
    Load any image (RGB, RGBA, L, …), convert to grayscale, resize to
    (size × size), and return a float32 array normalised to [0, 1].
    """
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parameter helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_L(K: float) -> float:
    """
    Compute the lower bound L for the sheet dynamic range [L, L+K].

    From paper Eq.(2), sufficient security condition:
        tT ∈ [max(0, 2(L+K)−1),  L]
    Rearranging for the tightest usable L that avoids CFR=0:
        L = max(0,  0.5 − K/2)

    With K = 0.688 (paper's example): L ≈ 0.156, so sheets ∈ [0.156, 0.844].
    Conflict Fulfillment Rate (CFR) from paper §4.2:
        CFR must be ≥ 0.6 for security (target info not visible in shares).
    """
    return float(max(0.0, 0.5 - K / 2.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core EVCS algorithm
# ─────────────────────────────────────────────────────────────────────────────

def run_evcs(
    cover1: np.ndarray,
    cover2: np.ndarray,
    secret: np.ndarray,
    K: float = 0.688,
    m: int = 4,
    seed: int = 42,
) -> tuple:
    """
    Run the (2,2)-EVCS encryption.

    Parameters
    ----------
    cover1, cover2 : (H, W) float32 in [0, 1]
        Cover (carrier) images. 1.0 = white, 0.0 = black.
    secret : (H, W) float32 in [0, 1]
        Secret image. 1.0 = white (transparent), 0.0 = black (opaque).
        USE HIGH-CONTRAST IMAGES (e.g. bold black text on white) for best results.
    K : float
        Contrast width.  Paper uses K = 0.688 with CFR = 0.702 as their best example.
        Higher K → better image quality but lower CFR (potential security leakage).
    m : int
        Pixel expansion (subpixels per pixel).  Must be a perfect square: 1, 4, 9, 16.
        m=4  → 2×2 block, output is 2× the input in each dimension.
        m=1  → no expansion; very low quality (no subpixel dithering possible).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    share1, share2 : (H*bs, W*bs) uint8 arrays, values 0 (black) or 255 (white)
    cfr            : float, Constraint Fulfillment Rate
    """
    random.seed(seed)

    assert int(m**0.5) ** 2 == m, \
        f"m={m} is not a perfect square. Use 1, 4, 9, or 16."

    bs = int(m**0.5)          # block size per side
    H, W = cover1.shape
    L = compute_L(K)

    # ── Step 1: Reduce dynamic ranges ──────────────────────────────────────
    # Sheets  : [L, L+K]   (paper §3.1, dynamic range constraint)
    # Target  : [0, K]
    t1_map = L + cover1 * K
    t2_map = L + cover2 * K
    tT_map = secret * K      # bright → transparent; dark → opaque

    # ── Error accumulator arrays (Floyd-Steinberg) ──────────────────────────
    e1 = np.zeros((H, W), np.float64)
    e2 = np.zeros((H, W), np.float64)
    eT = np.zeros((H, W), np.float64)

    # ── Output share arrays ─────────────────────────────────────────────────
    sh1 = np.zeros((H * bs, W * bs), np.uint8)
    sh2 = np.zeros((H * bs, W * bs), np.uint8)

    def diffuse(buf, y, x, err):
        """Floyd-Steinberg 4-neighbour error diffusion."""
        if x + 1 < W:
            buf[y, x + 1]     += err * 7.0 / 16.0
        if y + 1 < H:
            if x > 0:
                buf[y + 1, x - 1] += err * 3.0 / 16.0
            buf[y + 1, x]         += err * 5.0 / 16.0
            if x + 1 < W:
                buf[y + 1, x + 1] += err * 1.0 / 16.0

    conflicts = 0

    # ── Main pixel loop ─────────────────────────────────────────────────────
    for y in range(H):
        for x in range(W):

            # Apply accumulated error to each channel
            v1 = float(np.clip(t1_map[y, x] + e1[y, x], 0.0, 1.0))
            v2 = float(np.clip(t2_map[y, x] + e2[y, x], 0.0, 1.0))
            vT = float(np.clip(tT_map[y, x] + eT[y, x], 0.0, 1.0))

            # Quantise to integer subpixel counts in {0, …, m}
            s1 = int(round(v1 * m))
            s2 = int(round(v2 * m))
            sT = int(round(vT * m))
            s1 = max(0, min(m, s1))
            s2 = max(0, min(m, s2))
            sT = max(0, min(m, sT))

            # ── Check EVCS constraint Eq.(1) ─────────────────────────────
            #   sT ∈ [max(0, s1+s2−m),  min(s1, s2)]
            lo = max(0, s1 + s2 - m)
            hi = min(s1, s2)

            # Project sT into valid range if violated (paper §4.1)
            if sT < lo:
                conflicts += 1
                sT = lo
            elif sT > hi:
                conflicts += 1
                sT = hi

            # ── Subpixel pair counts (Table 1 in paper) ──────────────────
            #   P11 = both transparent  (contributes to target transparency)
            #   P10 = sh1 transparent, sh2 opaque
            #   P01 = sh1 opaque,      sh2 transparent
            #   P00 = both opaque
            P11 = sT
            P10 = s1 - sT
            P01 = s2 - sT
            P00 = m - s1 - s2 + sT

            # ── Build and randomly permute the m subpixel columns ────────
            #   row1[i] = transparency of subpixel i in share 1
            #   row2[i] = transparency of subpixel i in share 2
            row1 = [1] * P11 + [1] * P10 + [0] * P01 + [0] * P00
            row2 = [1] * P11 + [0] * P10 + [1] * P01 + [0] * P00
            perm = list(range(m))
            random.shuffle(perm)
            row1 = [row1[i] for i in perm]
            row2 = [row2[i] for i in perm]

            # ── Write subpixels into bs×bs block in output arrays ────────
            for k in range(m):
                by = y * bs + k // bs
                bx = x * bs + k % bs
                sh1[by, bx] = 255 if row1[k] else 0
                sh2[by, bx] = 255 if row2[k] else 0

            # ── Propagate quantisation errors ────────────────────────────
            diffuse(e1, y, x, v1 - s1 / m)
            diffuse(e2, y, x, v2 - s2 / m)
            diffuse(eT, y, x, vT - sT / m)

    cfr = 1.0 - conflicts / max(1, H * W)
    return sh1, sh2, cfr


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stacking simulation
# ─────────────────────────────────────────────────────────────────────────────

def stack_shares(sh1: np.ndarray, sh2: np.ndarray) -> np.ndarray:
    """
    Simulate physically stacking two printed transparencies.

    Transparency model:  255 = transparent (white),  0 = opaque (black).
    Light passes only where BOTH are transparent → AND = pixel-wise MIN.
    """
    return np.minimum(sh1, sh2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Built-in demo image generators (no external files required)
# ─────────────────────────────────────────────────────────────────────────────

def demo_cover1(size: int) -> np.ndarray:
    """
    Simulated natural photo (cover 1): dark sky + bright sun + terrain.
    Full dynamic range [0, 255] so the cover has both dark and bright areas,
    which is essential for good CFR.
    """
    rng = np.random.RandomState(1)
    arr = np.zeros((size, size), np.float32)

    # Sky (dark-to-mid gradient at top)
    for y in range(size // 2):
        arr[y, :] = 30 + 80 * (y / (size // 2))

    # Terrain (lighter at bottom)
    for y in range(size // 2, size):
        arr[y, :] = 120 + 60 * ((y - size // 2) / (size // 2))

    # Sun (bright circle)
    cx, cy = int(size * 0.7), int(size * 0.2)
    Y, X = np.ogrid[:size, :size]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    arr[r < size * 0.12] = 240

    # Clouds (white blobs)
    for cx_, cy_, rad in [(int(size*0.25), int(size*0.15), int(size*0.1)),
                          (int(size*0.45), int(size*0.1),  int(size*0.08))]:
        r2 = np.sqrt((X - cx_) ** 2 + (Y - cy_) ** 2)
        arr[r2 < rad] = np.clip(arr[r2 < rad] + 140, 0, 255)

    # Subtle noise for naturalism
    arr += rng.randn(size, size) * 8
    return arr.clip(0, 255).astype(np.uint8)


def demo_cover2(size: int) -> np.ndarray:
    """
    Simulated portrait-like image (cover 2): face silhouette on light background.
    Full dynamic range ensures good EVCS performance.
    """
    rng = np.random.RandomState(2)
    arr = np.full((size, size), 210, np.float32)   # light background

    # Oval face shape (darker)
    Y, X = np.ogrid[:size, :size]
    cx, cy = size // 2, int(size * 0.45)
    face = ((X - cx) / (size * 0.28)) ** 2 + ((Y - cy) / (size * 0.35)) ** 2 < 1.0
    arr[face] = 160

    # Dark hair at top
    hair = (Y < int(size * 0.35)) & face
    arr[hair] = 40

    # Eyes (very dark)
    for ex in [int(size*0.37), int(size*0.63)]:
        ey = int(size * 0.42)
        eye = np.sqrt((X - ex)**2 + (Y - ey)**2) < size * 0.04
        arr[eye] = 20

    # Shoulder area (dark clothing)
    arr[int(size*0.78):, :] = 55

    arr += rng.randn(size, size) * 6
    return arr.clip(0, 255).astype(np.uint8)


def demo_secret(size: int) -> np.ndarray:
    """
    Purely binary secret (black on white) — gives the best EVCS visibility.
    Uses a padlock symbol + text.
    """
    img = Image.new("L", (size, size), 255)   # white background
    draw = ImageDraw.Draw(img)

    # Outer border
    bw = max(2, size // 40)
    draw.rectangle([bw, bw, size - bw - 1, size - bw - 1], outline=0, width=bw)

    # Padlock body (filled rectangle)
    bx0 = int(size * 0.32)
    bx1 = int(size * 0.68)
    by0 = int(size * 0.48)
    by1 = int(size * 0.78)
    draw.rectangle([bx0, by0, bx1, by1], fill=0)

    # Padlock shackle (thick arc = two vertical lines + top horizontal)
    sw = max(3, size // 20)           # shackle width
    sx0, sx1 = int(size*0.38), int(size*0.62)
    sy0 = int(size * 0.28)
    sy1 = int(size * 0.54)
    draw.rectangle([sx0, sy0, sx0 + sw, sy1], fill=0)
    draw.rectangle([sx1 - sw, sy0, sx1, sy1], fill=0)
    draw.rectangle([sx0, sy0, sx1, sy0 + sw], fill=0)

    # Keyhole (white cutout in padlock body)
    kx = size // 2
    ky = int(size * 0.60)
    kr = max(3, size // 18)
    draw.ellipse([kx - kr, ky - kr, kx + kr, ky + kr], fill=255)
    draw.rectangle([kx - kr // 2, ky, kx + kr // 2, ky + int(size * 0.12)], fill=255)

    # "SECRET" text below
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, size=max(10, size // 7))
            break
        except Exception:
            pass

    text = "SECRET"
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
    else:
        tw = len(text) * 6
    draw.text(((size - tw) / 2, int(size * 0.83)), text, fill=0, font=font)

    return np.array(img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_array(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_composite(
    cover1_arr, cover2_arr, secret_arr,
    sh1, sh2, stacked,
    path: str,
) -> None:
    """
    Save a 2×3 grid PNG showing all six images with labels.
    Row 0: cover1 | cover2 | secret
    Row 1: share1 | share2 | stacked result
    """
    def to_pil(arr):
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")

    imgs_top = [cover1_arr, cover2_arr, secret_arr]
    imgs_bot = [sh1, sh2, stacked]
    labels_top = ["Cover 1", "Cover 2", "Secret (input)"]
    labels_bot = ["Share 1 (distribute)", "Share 2 (distribute)", "Stacked = Secret revealed"]

    n = len(imgs_top)
    H, W = sh1.shape
    pad = 4
    label_h = 22
    cell_w = W + pad * 2
    cell_h = H + pad * 2 + label_h

    total_w = cell_w * n
    total_h = cell_h * 2

    composite = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for col, (img_arr, label) in enumerate(zip(imgs_top, labels_top)):
        pil = to_pil(img_arr)
        pil = pil.resize((W, H), Image.LANCZOS)
        x = col * cell_w + pad
        y = pad + label_h
        composite.paste(pil, (x, y))
        draw.text((x, pad), label, fill=(50, 50, 50), font=font)

    for col, (img_arr, label) in enumerate(zip(imgs_bot, labels_bot)):
        pil = to_pil(img_arr)
        x = col * cell_w + pad
        y = cell_h + pad + label_h
        composite.paste(pil, (x, y))
        draw.text((x, cell_h + pad), label, fill=(50, 50, 50), font=font)

    composite.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extended Visual Cryptography Scheme (EVCS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMAGE GUIDE (read this before giving inputs):
  ┌──────────────┬──────────────────────────────────────────────────────────┐
  │ Image        │ What works well                                          │
  ├──────────────┼──────────────────────────────────────────────────────────┤
  │ --cover1     │ Any natural photo. Grayscale or colour (auto-converted). │
  │              │ Avoid pure black/white; mid-tone photos are ideal.        │
  │ --cover2     │ Same as cover1. Use a DIFFERENT image from cover1.        │
  │ --secret     │ HIGH CONTRAST only — bold black text on white, simple    │
  │              │ logos, QR codes, silhouettes. The secret is encoded via   │
  │              │ subpixel dithering; low-contrast photos will be invisible │
  │              │ in the stacked output.                                     │
  └──────────────┴──────────────────────────────────────────────────────────┘
  Size: all images are auto-resized to --size × --size. Recommended: 128 or 256.
  Format: PNG, JPEG, BMP — anything PIL can open.

PARAMETER GUIDE:
  --K     0.688   Paper's best result (CFR=0.702). Range: 0.3–0.9.
                  Higher K → clearer secret but more bleed into shares.
  --m     4       2×2 subpixel block. Output is 2× input size.
                  Use 9 (3×3) for better quality at the cost of 3× size.
  --size  128     128 = fast & fine. 256 = better quality, ~4× slower.

EXAMPLE:
  python evcs.py --cover1 face.jpg --cover2 landscape.jpg --secret qr.png
""",
    )
    p.add_argument("--cover1", default=None, help="Cover image for share 1")
    p.add_argument("--cover2", default=None, help="Cover image for share 2")
    p.add_argument("--secret", default=None, help="Secret image")
    p.add_argument("--size",   type=int,   default=128,   help="Resize all images to SIZE×SIZE (default 128)")
    p.add_argument("--K",      type=float, default=0.688, help="Contrast K ∈ (0,1] (default 0.688)")
    p.add_argument("--m",      type=int,   default=4,     help="Pixel expansion m, must be perfect square (default 4)")
    p.add_argument("--out",    default="evcs_output",     help="Output directory (default ./evcs_output)")
    p.add_argument("--seed",   type=int,   default=42,    help="Random seed (default 42)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate m ────────────────────────────────────────────────────────
    if int(args.m ** 0.5) ** 2 != args.m:
        print(f"ERROR: --m {args.m} is not a perfect square. Use 1, 4, 9, or 16.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    bs   = int(args.m ** 0.5)
    L    = compute_L(args.K)
    size = args.size

    print("=" * 62)
    print("  Extended Visual Cryptography Scheme (EVCS)")
    print("  Nakajima & Yamaguchi 2002  +  Liu & Wu 2011")
    print("=" * 62)
    print(f"  Input size      : {size}×{size} px")
    print(f"  Output size     : {size*bs}×{size*bs} px  (pixel expansion m={args.m})")
    print(f"  Contrast K      : {args.K}")
    print(f"  Lower bound L   : {L:.4f}")
    print(f"  Sheet range     : [{L:.3f}, {L+args.K:.3f}]")
    print(f"  Target range    : [0.000, {args.K:.3f}]")
    print(f"  Random seed     : {args.seed}")
    print()

    # ── Load images ───────────────────────────────────────────────────────
    if args.cover1:
        print(f"[1/3] Loading cover1  : {args.cover1}")
        c1_arr = (load_image(args.cover1, size) * 255).astype(np.uint8)
        c1     = c1_arr.astype(np.float32) / 255.0
    else:
        print("[1/3] No --cover1 given — using built-in demo (concentric rings).")
        c1_arr = demo_cover1(size)
        Image.fromarray(c1_arr).save(os.path.join(args.out, "demo_cover1.png"))
        c1 = c1_arr.astype(np.float32) / 255.0

    if args.cover2:
        print(f"[2/3] Loading cover2  : {args.cover2}")
        c2_arr = (load_image(args.cover2, size) * 255).astype(np.uint8)
        c2     = c2_arr.astype(np.float32) / 255.0
    else:
        print("[2/3] No --cover2 given — using built-in demo (diagonal gradient).")
        c2_arr = demo_cover2(size)
        Image.fromarray(c2_arr).save(os.path.join(args.out, "demo_cover2.png"))
        c2 = c2_arr.astype(np.float32) / 255.0

    if args.secret:
        print(f"[3/3] Loading secret  : {args.secret}")
        s_arr = (load_image(args.secret, size) * 255).astype(np.uint8)
        sec   = s_arr.astype(np.float32) / 255.0
    else:
        print("[3/3] No --secret given — using built-in 'SECRET MSG' text demo.")
        s_arr = demo_secret(size)
        Image.fromarray(s_arr).save(os.path.join(args.out, "demo_secret.png"))
        sec = s_arr.astype(np.float32) / 255.0

    # ── Run EVCS ──────────────────────────────────────────────────────────
    print()
    print("Running EVCS encryption (this may take a few seconds)…")
    sh1, sh2, cfr = run_evcs(c1, c2, sec, K=args.K, m=args.m, seed=args.seed)

    # ── Security assessment ───────────────────────────────────────────────
    print(f"  Constraint Fulfillment Rate (CFR) : {cfr:.4f}")
    if cfr >= 0.7:
        print("  Security assessment : GOOD  (CFR ≥ 0.70, paper benchmark)")
    elif cfr >= 0.6:
        print("  Security assessment : ACCEPTABLE  (CFR ≥ 0.60, paper threshold)")
    else:
        print("  Security assessment : POOR  (CFR < 0.60 — secret may bleed into shares)")
        print("  Suggestion: lower K (e.g. --K 0.5) to reduce conflicts.")

    stacked = stack_shares(sh1, sh2)

    # ── Save outputs ──────────────────────────────────────────────────────
    p_sh1     = os.path.join(args.out, "share1.png")
    p_sh2     = os.path.join(args.out, "share2.png")
    p_stacked = os.path.join(args.out, "stacked_result.png")
    p_grid    = os.path.join(args.out, "composite_all.png")

    save_array(sh1,     p_sh1)
    save_array(sh2,     p_sh2)
    save_array(stacked, p_stacked)
    save_composite(c1_arr, c2_arr, s_arr, sh1, sh2, stacked, p_grid)

    print()
    print("Output files:")
    print(f"  {p_sh1}")
    print(f"  {p_sh2}")
    print(f"  {p_stacked}")
    print(f"  {p_grid}  ← side-by-side composite of all 6 images")
    print()


if __name__ == "__main__":
    main()
