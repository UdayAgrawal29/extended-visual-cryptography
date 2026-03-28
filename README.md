# Visual Cryptography (Grayscale)

This repository contains a grayscale implementation of Extended Visual Cryptography Scheme (EVCS) using meaningful cover images.

Current status:
- Grayscale EVCS is implemented and working in `evcs.py`.
- RGB version is planned next.

## Important file notes

- `evcs.py`: current and working implementation.
- `evc.py`: older implementation with failed/incorrect output (kept for reference).


## Papers used as reference

This implementation is based on:
- Nakajima and Yamaguchi (2002), Extended Visual Cryptography for Natural Images
- Liu and Wu (2011), Embedded Extended Visual Cryptography Schemes

## Project structure

- `covers/`: input cover images (share 1 and share 2 inputs)
- `secret/`: input secret image
- `evcs_output/`: output from `evcs.py`
- `evc_output/`: output from `evc.py`

## Requirements

Install dependencies:

```bash
pip install numpy Pillow
```

## Run EVCS

### Command for current folder structure

```bash
python evcs.py --cover1 covers/c1.jpg --cover2 covers/c3.jpg --secret secret/s2.jpg --size 256 --m 16
```

You can change any parameter as needed.

## Parameter guide

- `--cover1`: path to cover image used to generate share 1.
- `--cover2`: path to cover image used to generate share 2.
- `--secret`: path to secret image revealed after stacking both shares.
- `--size`: resize all three input images to `size x size` before processing.
- `--m`: pixel expansion (must be a perfect square like 1, 4, 9, 16).
  - Example: `m=16` means a 4x4 subpixel block per input pixel.
- `--K`: contrast control (default `0.688` in code).
  - Higher `K`: stronger visual contrast, can reduce security margin.
- `--seed`: random seed for reproducible share generation.
- `--out`: output directory (default `evcs_output`).

## Outputs

By default, `evcs.py` saves:
- `evcs_output/share1.png`
- `evcs_output/share2.png`
- `evcs_output/stacked_result.png`
- `evcs_output/composite_all.png`

