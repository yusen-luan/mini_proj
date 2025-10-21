import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _save_image(path: Path, image: np.ndarray) -> None:
    _ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


def _random_degree(a: float, b: float) -> float:
    return random.uniform(a, b)


def _random_sign() -> int:
    return -1 if random.random() < 0.5 else 1


def _rotation(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _scale(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    mat = np.array([[scale, 0, (1 - scale) * w / 2], [0, scale, (1 - scale) * h / 2]], dtype=np.float32)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _translate(image: np.ndarray, tx: float, ty: float) -> np.ndarray:
    h, w = image.shape[:2]
    mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _shear(image: np.ndarray, shear_deg_x: float, shear_deg_y: float) -> np.ndarray:
    h, w = image.shape[:2]
    shx = math.tan(math.radians(shear_deg_x))
    shy = math.tan(math.radians(shear_deg_y))
    mat = np.array([[1, shx, -shx * w / 2], [shy, 1, -shy * h / 2]], dtype=np.float32)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _perspective(image: np.ndarray, strength: float) -> np.ndarray:
    h, w = image.shape[:2]
    dx = strength * w
    dy = strength * h

    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [0 + random.uniform(-dx, dx), 0 + random.uniform(-dy, dy)],
            [w - 1 + random.uniform(-dx, dx), 0 + random.uniform(-dy, dy)],
            [w - 1 + random.uniform(-dx, dx), h - 1 + random.uniform(-dy, dy)],
            [0 + random.uniform(-dx, dx), h - 1 + random.uniform(-dy, dy)],
        ]
    )
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def geom_aug(
    input_dir: Path = Path("relabelled_yusen"), output_root: Path = Path("augmented")
) -> None:
    """
    Perform geometric augmentations on each image in input_dir and save results into
    output_root/geom/{rotation,scale,translation,shear,perspective}/ with the same filenames.

    Augmentations (randomized within ranges per image):
      - Rotation: ±(5–12) degrees
      - Scale: 0.9–1.1 (keep aspect ratio)
      - Translation: up to ±6% width/height with padding
      - Shear: ±(6–10) degrees (both axes independently)
      - Perspective warp: 0.02–0.07
    """

    input_dir = Path(input_dir)
    output_root = Path(output_root)
    out_geom = output_root / "geom"
    out_dirs = {
        "rotation": out_geom / "rotate",
        "scale": out_geom / "scale",
        "translation": out_geom / "translate",
        "shear": out_geom / "shear",
        "perspective": out_geom / "persp_warp",
    }
    for p in out_dirs.values():
        _ensure_dir(p)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(input_dir.iterdir(), key=lambda q: q.name) if p.is_file() and p.suffix.lower() in exts]

    for img_path in files:
        img = _load_image(img_path)
        h, w = img.shape[:2]

        # Rotation: ±(5–12)°
        rot_deg = _random_sign() * _random_degree(5.0, 12.0)
        img_rot = _rotation(img, rot_deg)
        _save_image(out_dirs["rotation"] / img_path.name, img_rot)

        # Scale/Zoom: 0.9–1.1; keep aspect ratio
        sc = random.uniform(0.9, 1.1)
        img_scale = _scale(img, sc)
        _save_image(out_dirs["scale"] / img_path.name, img_scale)

        # Translation: up to ±6% width/height
        tx = random.uniform(-0.06, 0.06) * w
        ty = random.uniform(-0.06, 0.06) * h
        img_trans = _translate(img, tx, ty)
        _save_image(out_dirs["translation"] / img_path.name, img_trans)

        # Shear: ±6–10° (independent X and Y)
        shear_x = _random_sign() * _random_degree(6.0, 10.0)
        shear_y = _random_sign() * _random_degree(6.0, 10.0)
        img_shear = _shear(img, shear_x, shear_y)
        _save_image(out_dirs["shear"] / img_path.name, img_shear)

        # Perspective warp: 0.02–0.07
        persp_strength = random.uniform(0.02, 0.07)
        img_persp = _perspective(img, persp_strength)
        _save_image(out_dirs["perspective"] / img_path.name, img_persp)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    # Let OpenCV choose kernel size from sigma by providing ksize=(0, 0)
    return cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def _motion_blur(image: np.ndarray, kernel_size: int, direction: str) -> np.ndarray:
    # direction: 'horizontal' or 'vertical'
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
    else:
        kernel[:, kernel_size // 2] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _defocus_blur(image: np.ndarray, radius: int) -> np.ndarray:
    # Disk kernel with given radius
    k = 2 * radius + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    mask = (xx * xx + yy * yy) <= (radius * radius)
    kernel = mask.astype(np.float32)
    s = kernel.sum()
    if s <= 0:
        kernel = np.ones((k, k), dtype=np.float32) / float(k * k)
    else:
        kernel /= s
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def degrade_aug(
    input_dir: Path = Path("relabelled_yusen"), output_root: Path = Path("augmented")
) -> None:
    """
    Apply degradation augmentations (gaussian, motion, defocus blurs) to each image in input_dir
    and save into output_root/degrade/{gaussian,motion,defocus}/ with original filenames.

    Ranges:
      - Gaussian blur: sigma 0.3–1.2
      - Motion blur: kernel size k in {3,5,7}; direction random (horizontal/vertical)
      - Defocus blur: small radius (2–3)
    """

    input_dir = Path(input_dir)
    output_root = Path(output_root)
    out_deg = output_root / "degrade"
    out_dirs = {
        "gaussian": out_deg / "gaussian",
        "motion": out_deg / "motion",
        "defocus": out_deg / "defocus",
    }
    for p in out_dirs.values():
        _ensure_dir(p)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(input_dir.iterdir(), key=lambda q: q.name) if p.is_file() and p.suffix.lower() in exts]

    for img_path in files:
        img = _load_image(img_path)

        # Gaussian blur
        sigma = random.uniform(0.3, 1.2)
        img_g = _gaussian_blur(img, sigma)
        _save_image(out_dirs["gaussian"] / img_path.name, img_g)

        # Motion blur
        k = random.choice([3, 5, 7])
        direction = "horizontal" if random.random() < 0.5 else "vertical"
        img_m = _motion_blur(img, k, direction)
        _save_image(out_dirs["motion"] / img_path.name, img_m)

        # Defocus blur
        radius = random.choice([2, 3])
        img_d = _defocus_blur(img, radius)
        _save_image(out_dirs["defocus"] / img_path.name, img_d)


def _apply_brightness_contrast_gamma(
    image: np.ndarray,
    brightness_pct_range: tuple[float, float] = (0.15, 0.25),
    contrast_pct_range: tuple[float, float] = (0.15, 0.25),
    gamma_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    # Contrast factor around 1.0 with ±jitter
    contrast_delta = random.uniform(contrast_pct_range[0], contrast_pct_range[1])
    contrast_factor = 1.0 + (contrast_delta if random.random() < 0.5 else -contrast_delta)

    # Brightness offset as ±% of 255
    brightness_delta = random.uniform(brightness_pct_range[0], brightness_pct_range[1])
    brightness_shift = (brightness_delta if random.random() < 0.5 else -brightness_delta) * 255.0

    x = image.astype(np.float32)
    x = x * contrast_factor + brightness_shift
    x = np.clip(x, 0, 255)

    # Gamma correction via LUT
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    inv = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.float32)
    x = cv2.LUT(x.astype(np.uint8), lut.astype(np.uint8))
    return x


def _jitter_saturation_hue(
    image: np.ndarray,
    sat_pct_range: tuple[float, float] = (0.0, 0.2),
    hue_deg_range: tuple[float, float] = (0.0, 5.0),
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.float32)

    # Saturation multiplier
    sat_delta = random.uniform(sat_pct_range[0], sat_pct_range[1])
    sat_factor = 1.0 + (sat_delta if random.random() < 0.5 else -sat_delta)
    s = np.clip(s * sat_factor, 0, 255)

    # Hue shift in degrees; OpenCV hue scale is 0..179 ~ 0..360 degrees
    hue_deg = random.uniform(hue_deg_range[0], hue_deg_range[1])
    hue_shift = hue_deg if random.random() < 0.5 else -hue_deg
    hue_units = int(round(hue_shift * (180.0 / 360.0)))  # degrees to OpenCV units
    h = (h + hue_units) % 180

    hsv[:, :, 0] = h.astype(np.uint8)
    hsv[:, :, 1] = s.astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def _to_grayscale3(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _invert(image: np.ndarray) -> np.ndarray:
    return 255 - image


def photo_aug(
    input_dir: Path = Path("relabelled_yusen"), output_root: Path = Path("augmented")
) -> None:
    """
    Apply photometric augmentations and save into output_root/photo/{bcg,sat_hue,grayscale,invert}/.

    - Brightness/Contrast/Gamma jitter: brightness ±15–25% (of 255), contrast ±15–25%, gamma 0.8–1.2
    - Saturation/Hue jitter: saturation ±20%, hue ±5°
    - Grayscale conversion
    - Invert colors
    """

    input_dir = Path(input_dir)
    output_root = Path(output_root)
    out_photo = output_root / "photo"
    out_dirs = {
        "bcg": out_photo / "bcg",
        "sat_hue": out_photo / "sat_hue",
        "grayscale": out_photo / "grayscale",
        "invert": out_photo / "invert",
    }
    for p in out_dirs.values():
        _ensure_dir(p)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(input_dir.iterdir(), key=lambda q: q.name) if p.is_file() and p.suffix.lower() in exts]

    for img_path in files:
        img = _load_image(img_path)

        img_bcg = _apply_brightness_contrast_gamma(img)
        _save_image(out_dirs["bcg"] / img_path.name, img_bcg)

        img_sh = _jitter_saturation_hue(img)
        _save_image(out_dirs["sat_hue"] / img_path.name, img_sh)

        img_gray = _to_grayscale3(img)
        _save_image(out_dirs["grayscale"] / img_path.name, img_gray)

        img_inv = _invert(img)
        _save_image(out_dirs["invert"] / img_path.name, img_inv)


def _lines_and_arcs(
    image: np.ndarray, num_strokes_range: tuple[int, int] = (1, 3), thickness_range: tuple[int, int] = (1, 2)
) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    num_strokes = random.randint(num_strokes_range[0], num_strokes_range[1])
    for _ in range(num_strokes):
        thickness = random.randint(thickness_range[0], thickness_range[1])
        color_val = random.randint(20, 235)
        color = (color_val, color_val, color_val)
        if random.random() < 0.5:
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
            cv2.line(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        else:
            center = (random.randint(0, w - 1), random.randint(0, h - 1))
            axes = (random.randint(w // 6, w // 3), random.randint(h // 6, h // 3))
            angle = random.uniform(0, 360)
            start_angle = random.uniform(0, 360)
            end_angle = (start_angle + random.uniform(30, 180)) % 360
            cv2.ellipse(out, center, axes, angle, start_angle, end_angle, color, thickness, lineType=cv2.LINE_AA)
    return out


def _dots_splatter(
    image: np.ndarray, num_dots_range: tuple[int, int] = (5, 50), radius_range: tuple[int, int] = (1, 3)
) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    num_dots = random.randint(num_dots_range[0], num_dots_range[1])
    for _ in range(num_dots):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        r = random.randint(radius_range[0], radius_range[1])
        color_val = random.randint(0, 255)
        color = (color_val, color_val, color_val)
        cv2.circle(out, (x, y), r, color, thickness=-1, lineType=cv2.LINE_AA)
    return out


def _estimate_background_color(image: np.ndarray) -> tuple[int, int, int]:
    h, w = image.shape[:2]
    border = 5
    # Flatten each border region to (N, 3) before concatenation to avoid dimension mismatch
    strips = [
        image[0:border, :, :].reshape(-1, 3),
        image[h - border : h, :, :].reshape(-1, 3),
        image[:, 0:border, :].reshape(-1, 3),
        image[:, w - border : w, :].reshape(-1, 3),
    ]
    samples = np.concatenate(strips, axis=0)
    bg = np.median(samples, axis=0)
    return int(bg[0]), int(bg[1]), int(bg[2])


def _light_cutouts(image: np.ndarray, max_area_ratio: float = 0.10) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    max_area = max_area_ratio * float(h * w)
    used_area = 0.0
    bg_b, bg_g, bg_r = _estimate_background_color(image)
    attempt = 0
    while used_area < max_area and attempt < 50:
        attempt += 1
        rect_w = random.randint(max(2, w // 50), max(3, w // 20))
        rect_h = random.randint(max(2, h // 50), max(3, h // 20))
        if used_area + rect_w * rect_h > max_area:
            break
        x1 = random.randint(0, max(0, w - rect_w))
        y1 = random.randint(0, max(0, h - rect_h))
        jitter = lambda v: int(np.clip(v + random.randint(-10, 10), 0, 255))
        color = (jitter(bg_b), jitter(bg_g), jitter(bg_r))
        cv2.rectangle(out, (x1, y1), (x1 + rect_w, y1 + rect_h), color, thickness=-1)
        used_area += rect_w * rect_h
    return out


def _background_texture_overlay(image: np.ndarray, alpha_range: tuple[float, float] = (0.10, 0.25)) -> np.ndarray:
    h, w = image.shape[:2]
    # Base noise texture
    noise = np.random.normal(loc=127.0, scale=8.0, size=(h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.5)
    texture = np.dstack([noise, noise, noise])
    # Add subtle grid
    grid = np.zeros((h, w), dtype=np.float32)
    spacing = random.randint(12, 28)
    grid_color = random.uniform(110, 145)
    grid[::spacing, :] = grid_color
    grid[:, ::spacing] = grid_color
    grid = cv2.GaussianBlur(grid, (0, 0), 0.8)
    texture += np.dstack([grid, grid, grid])
    texture = np.clip(texture, 0, 255).astype(np.uint8)

    alpha = random.uniform(alpha_range[0], alpha_range[1])
    blended = cv2.addWeighted(image, 1.0 - alpha, texture, alpha, 0.0)
    return blended


def clutter_aug(
    input_dir: Path = Path("relabelled_yusen"), output_root: Path = Path("augmented")
) -> None:
    """
    Apply clutter augmentations and save into output_root/clutter/{lines_arcs,dots,cutouts,texture}/.
    """
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    out_clutter = output_root / "clutter"
    out_dirs = {
        "lines_arcs": out_clutter / "lines_arcs",
        "dots": out_clutter / "dots",
        "cutouts": out_clutter / "cutouts",
        "texture": out_clutter / "texture",
    }
    for p in out_dirs.values():
        _ensure_dir(p)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(input_dir.iterdir(), key=lambda q: q.name) if p.is_file() and p.suffix.lower() in exts]

    for img_path in files:
        img = _load_image(img_path)

        img_la = _lines_and_arcs(img)
        _save_image(out_dirs["lines_arcs"] / img_path.name, img_la)

        img_dots = _dots_splatter(img)
        _save_image(out_dirs["dots"] / img_path.name, img_dots)

        img_cut = _light_cutouts(img)
        _save_image(out_dirs["cutouts"] / img_path.name, img_cut)

        img_tex = _background_texture_overlay(img)
        _save_image(out_dirs["texture"] / img_path.name, img_tex)

def main() -> None:
    parser = argparse.ArgumentParser(description="Image augmentations: geometric, degrade, photometric, and clutter.")
    parser.add_argument(
        "--input",
        type=str,
        default="relabelled_yusen",
        help="Input folder containing images (default: relabelled_yusen)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="augmented",
        help="Output root folder (default: augmented)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["geom", "degrade", "photo", "clutter"],
        default="geom",
        help="Which augmentation pipeline to run (geom, degrade, photo, or clutter)",
    )
    args = parser.parse_args()

    if args.task == "geom":
        geom_aug(Path(args.input), Path(args.output))
    elif args.task == "degrade":
        degrade_aug(Path(args.input), Path(args.output))
    elif args.task == "photo":
        photo_aug(Path(args.input), Path(args.output))
    else:
        clutter_aug(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()


