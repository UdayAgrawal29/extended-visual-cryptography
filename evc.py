"""
Extended Visual Cryptography for Natural Images
Implementation of: Nakajima & Yamaguchi, University of Tokyo

Algorithm overview:
  - Takes 3 grayscale images: sheet1, sheet2, target
  - Outputs 2 encrypted sheets such that stacking them reveals the target
  - Uses Floyd-Steinberg error diffusion for simultaneous halftoning + encryption
  - Subpixel arrangement (m subpixels per pixel) controls transparency
"""

import numpy as np
from PIL import Image, ImageFilter
import argparse
import os
import sys


# ─────────────────────────────────────────────
#  Core Math
# ─────────────────────────────────────────────

def affine_transform(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """Scale a [0,1] image to [low, high]."""
    return img * (high - low) + low


def satisfies_condition1(s1: int, s2: int, sT: int, m: int) -> bool:
    """
    Condition (1) from the paper:
      tT ∈ [max(0, t1+t2-1), min(t1,t2)]
    In subpixel counts:
      sT ∈ [max(0, s1+s2-m), min(s1,s2)]
    """
    lo = max(0, s1 + s2 - m)
    hi = min(s1, s2)
    return lo <= sT <= hi


def project_to_valid(s1f: float, s2f: float, sTf: float, m: int):
    """
    Project a floating-point triplet to the nearest integer triplet
    that satisfies condition (1).  Returns (s1, s2, sT, err1, err2, errT).
    """
    # Round candidates within [0, m]
    best = None
    best_dist = float('inf')

    # Search small neighbourhood around the float values
    for ds1 in range(-1, 2):
        for ds2 in range(-1, 2):
            for dsT in range(-1, 2):
                s1 = int(np.clip(round(s1f) + ds1, 0, m))
                s2 = int(np.clip(round(s2f) + ds2, 0, m))
                sT = int(np.clip(round(sTf) + dsT, 0, m))
                if satisfies_condition1(s1, s2, sT, m):
                    dist = (s1f/m - s1/m)**2 + \
                           (s2f/m - s2/m)**2 + \
                           (sTf/m - sT/m)**2
                    if dist < best_dist:
                        best_dist = dist
                        best = (s1, s2, sT)

    if best is None:
        # Fallback: clamp sT to valid range with rounded s1, s2
        s1 = int(np.clip(round(s1f), 0, m))
        s2 = int(np.clip(round(s2f), 0, m))
        lo = max(0, s1 + s2 - m)
        hi = min(s1, s2)
        sT = int(np.clip(round(sTf), lo, hi))
        best = (s1, s2, sT)

    s1, s2, sT = best
    err1 = s1f/m - s1/m
    err2 = s2f/m - s2/m
    errT = sTf/m - sT/m
    return s1, s2, sT, err1, err2, errT


def arrange_subpixels(s1: int, s2: int, sT: int, m: int, rng: np.random.Generator):
    """
    Randomly arrange m subpixels for sheet1 and sheet2 such that
    the stacked transparency equals sT/m.

    Returns (row_sheet1, row_sheet2) each of length m,
    values in {0 (opaque), 1 (transparent)}.
    """
    P11 = sT           # both transparent → target transparent
    P10 = s1 - sT      # sheet1 transparent, sheet2 opaque
    P01 = s2 - sT      # sheet1 opaque, sheet2 transparent
    P00 = m - s1 - s2 + sT  # both opaque

    # Build column list and shuffle
    cols = ([[1, 1]] * P11 +
            [[1, 0]] * P10 +
            [[0, 1]] * P01 +
            [[0, 0]] * P00)
    cols = np.array(cols, dtype=np.uint8)
    rng.shuffle(cols)

    return cols[:, 0], cols[:, 1]


# ─────────────────────────────────────────────
#  Floyd-Steinberg Error Diffusion
# ─────────────────────────────────────────────

FS_WEIGHTS = [
    (0,  1, 7/16),
    (1, -1, 3/16),
    (1,  0, 5/16),
    (1,  1, 1/16),
]


def diffuse_error(buf: np.ndarray, y: int, x: int, err: float):
    h, w = buf.shape
    for dy, dx, w_ in FS_WEIGHTS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            buf[ny, nx] += err * w_


# ─────────────────────────────────────────────
#  Main Encryption
# ─────────────────────────────────────────────

def compute_optimal_L(K: float) -> float:
    """
    Analytically compute L that maximises the Constraint Fulfillment Rate
    under the assumption of uniform distribution (Section 4.3).
    For K <= 0.5 the optimum is L = (1-K)/2 (center the dynamic range).
    """
    return max(0.0, min((1.0 - K) / 2.0, 1.0 - K))


def encrypt(img_sheet1: np.ndarray,
            img_sheet2: np.ndarray,
            img_target: np.ndarray,
            K: float = 0.5,
            L: float = None,
            m: int = 4,
            seed: int = 42) -> tuple:
    """
    Encrypt three [0,1] grayscale images into two output sheets.

    Parameters
    ----------
    img_sheet1, img_sheet2 : ndarray [0,1]  — desired images on each sheet
    img_target             : ndarray [0,1]  — image revealed when sheets are stacked
    K                      : float          — contrast (dynamic range width)
    L                      : float|None     — lower bound of sheet dynamic range
                                              (auto-selected if None)
    m                      : int            — subpixels per pixel (must be perfect square)
    seed                   : int            — random seed for reproducibility

    Returns
    -------
    out_sheet1, out_sheet2 : uint8 ndarray  — encrypted sheets (h*sqrt_m, w*sqrt_m)
    cfr                    : float          — measured Constraint Fulfillment Rate
    """
    assert img_sheet1.shape == img_sheet2.shape == img_target.shape
    h, w = img_sheet1.shape
    sqrt_m = int(round(np.sqrt(m)))
    assert sqrt_m * sqrt_m == m, "m must be a perfect square (4, 9, 16, …)"

    if L is None:
        L = compute_optimal_L(K)

    # ── Affine-transform to target dynamic ranges ──
    t1_buf = affine_transform(img_sheet1, L, L + K).copy()
    t2_buf = affine_transform(img_sheet2, L, L + K).copy()
    tT_buf = affine_transform(img_target, 0.0, K).copy()

    # Error accumulation buffers
    err1 = np.zeros((h, w), dtype=float)
    err2 = np.zeros((h, w), dtype=float)
    errT = np.zeros((h, w), dtype=float)

    # Output images (white = transparent, black = opaque)
    out1 = np.zeros((h * sqrt_m, w * sqrt_m), dtype=np.uint8)
    out2 = np.zeros((h * sqrt_m, w * sqrt_m), dtype=np.uint8)

    rng = np.random.default_rng(seed)
    n_valid = 0
    n_total = h * w

    for y in range(h):
        for x in range(w):
            # Current pixel value + accumulated diffusion error
            t1 = np.clip(t1_buf[y, x] + err1[y, x], 0.0, 1.0)
            t2 = np.clip(t2_buf[y, x] + err2[y, x], 0.0, 1.0)
            tT = np.clip(tT_buf[y, x] + errT[y, x], 0.0, 1.0)

            # Project to nearest valid integer triplet (Section 4.1)
            s1, s2, sT, e1, e2, eT = project_to_valid(t1 * m, t2 * m, tT * m, m)

            if satisfies_condition1(s1, s2, sT, m):
                n_valid += 1

            # Diffuse quantisation errors
            diffuse_error(err1, y, x, e1)
            diffuse_error(err2, y, x, e2)
            diffuse_error(errT, y, x, eT)

            # Generate subpixel arrangements
            sp1, sp2 = arrange_subpixels(s1, s2, sT, m, rng)

            # Write subpixels into output images
            for i in range(sqrt_m):
                for j in range(sqrt_m):
                    idx = i * sqrt_m + j
                    out1[y * sqrt_m + i, x * sqrt_m + j] = sp1[idx] * 255
                    out2[y * sqrt_m + i, x * sqrt_m + j] = sp2[idx] * 255

    cfr = n_valid / n_total
    return out1, out2, cfr


# ─────────────────────────────────────────────
#  Stacking Simulation
# ─────────────────────────────────────────────

def simulate_stack(sheet1: np.ndarray, sheet2: np.ndarray) -> np.ndarray:
    """
    Simulate placing sheet2 on top of sheet1 on a transparency:
    a pixel is transparent (white=255) only if BOTH sheets are transparent.
    This is the AND of two binary images.
    """
    b1 = (sheet1 > 128).astype(np.uint8)
    b2 = (sheet2 > 128).astype(np.uint8)
    stacked = (b1 & b2).astype(np.uint8) * 255
    return stacked


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def load_image(path: str, size: tuple) -> np.ndarray:
    img = Image.open(path).convert('L').resize(size, Image.LANCZOS)
    return np.array(img, dtype=float) / 255.0


def save_image(arr: np.ndarray, path: str):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def make_demo_images(size=(64, 64)):
    """Generate synthetic test images when no real images are provided."""
    h, w = size
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    # Sheet 1: horizontal gradient with a circle
    img1 = np.clip(0.3 + 0.5 * xx + 0.1 * np.sin(10 * np.pi * yy), 0, 1)
    mask1 = ((xx - 0.5)**2 + (yy - 0.5)**2) < 0.08
    img1[mask1] = 0.8

    # Sheet 2: vertical gradient with a square
    img2 = np.clip(0.3 + 0.5 * yy + 0.1 * np.cos(8 * np.pi * xx), 0, 1)
    mask2 = (np.abs(xx - 0.5) < 0.2) & (np.abs(yy - 0.25) < 0.1)
    img2[mask2] = 0.15

    # Target: simple cross / smiley pattern
    imgT = np.full((h, w), 0.3)
    imgT[h//4:3*h//4, w//2-2:w//2+2] = 0.9  # vertical bar
    imgT[h//2-2:h//2+2, w//4:3*w//4] = 0.9  # horizontal bar

    return img1, img2, imgT


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extended Visual Cryptography for Natural Images"
    )
    parser.add_argument("--sheet1",  default=None, help="Path to sheet1 image")
    parser.add_argument("--sheet2",  default=None, help="Path to sheet2 image")
    parser.add_argument("--target",  default=None, help="Path to target image")
    parser.add_argument("--size",    type=int, default=128,
                        help="Resize images to SIZE×SIZE (default: 128)")
    parser.add_argument("--K",       type=float, default=0.6,
                        help="Contrast K ∈ (0,1] (default: 0.6)")
    parser.add_argument("--L",       type=float, default=None,
                        help="Lower bound L (auto if omitted)")
    parser.add_argument("--m",       type=int,   default=4,
                        help="Subpixels per pixel, must be perfect square (default: 4)")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--outdir",  default=r"C:\Users\Uday Agrawal\Downloads\New folder (7)",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    size = (args.size, args.size)

    # ── Load or generate images ──
    print(f"Loading images …")
    img1 = load_image(r"C:\Users\Uday Agrawal\Desktop\Projects\crypto\images\covers\cover3.jpg", size)
    img2 = load_image(r"C:\Users\Uday Agrawal\Desktop\Projects\crypto\images\covers\cover4.jpg", size)
    imgT = load_image(r"C:\Users\Uday Agrawal\Desktop\Projects\crypto\images\secret\secret2.jpg", size)

    # Save the (affine-transformed) inputs for reference
    save_image((img1 * 255).astype(np.uint8),
               os.path.join(args.outdir, "input_sheet1.png"))
    save_image((img2 * 255).astype(np.uint8),
               os.path.join(args.outdir, "input_sheet2.png"))
    save_image((imgT * 255).astype(np.uint8),
               os.path.join(args.outdir, "input_target.png"))

    # ── Encrypt ──
    print(f"\nEncrypting  K={args.K}  m={args.m}  size={size} …")
    L = args.L if args.L is not None else compute_optimal_L(args.K)
    print(f"  L = {L:.4f}  (sheet dynamic range [{L:.3f}, {L+args.K:.3f}])")

    out1, out2, cfr = encrypt(img1, img2, imgT,
                              K=args.K, L=L, m=args.m, seed=args.seed)

    print(f"  Constraint Fulfillment Rate (CFR) = {cfr:.4f}")
    if cfr < 0.6:
        print("  ⚠  CFR < 0.6 — target image may bleed through the sheets."
              " Try reducing K.")

    # ── Save outputs ──
    p1 = os.path.join(args.outdir, "encrypted_sheet1.png")
    p2 = os.path.join(args.outdir, "encrypted_sheet2.png")
    ps = os.path.join(args.outdir, "stacked_result.png")

    save_image(out1, p1)
    save_image(out2, p2)

    stacked = simulate_stack(out1, out2)
    save_image(stacked, ps)

    print(f"\nOutputs written to {args.outdir}/")
    print(f"  encrypted_sheet1.png  — sheet 1 (print on transparency)")
    print(f"  encrypted_sheet2.png  — sheet 2 (print on transparency)")
    print(f"  stacked_result.png    — simulated stack (reveals target)")
    print(f"  input_sheet1/2/target — input images for comparison")


if __name__ == "__main__":
    main()
