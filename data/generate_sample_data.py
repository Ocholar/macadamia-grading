"""
Sample Data and Synthetic Batch Image Generator.
Generates photorealistic macadamia batch images on standardized inspection mats,
along with ground-truth COCO annotations and segmented crops.
"""

import json
import os
import random
import logging
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_batch(
    output_path: str,
    num_nuts: int = 18,
    mat_color: str = "blue",
    defect_ratios: Tuple[float, float, float, float] = (0.70, 0.10, 0.10, 0.10),
    px_per_mm: float = 6.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates a realistic photo of a batch of macadamia nuts on an inspection mat.

    defect_ratios: (healthy, cracked, discolored, moldy)
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    w, h = 900, 900

    # 1. Base Mat Background
    if mat_color == "blue":
        # Standard blue rubber inspection mat with subtle texture
        mat_base = np.full((h, w, 3), [170, 115, 45], dtype=np.uint8)  # BGR
    elif mat_color == "green":
        mat_base = np.full((h, w, 3), [55, 120, 45], dtype=np.uint8)
    else:
        mat_base = np.full((h, w, 3), [60, 60, 60], dtype=np.uint8)

    # Add mat texture noise
    noise = np_rng.normal(0, 4, (h, w, 3)).astype(np.int16)
    mat = np.clip(mat_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Distribute Nuts without excessive overlap
    placed_nuts = []
    annotations = []

    classes_pool = [0, 1, 2, 3]

    for i in range(num_nuts):
        # Choose defect class based on distribution
        defect_class = rng.choices(classes_pool, weights=defect_ratios, k=1)[0]

        # Choose size in mm (Small: 15-17.5, Medium: 18.5-21.5, Large: 22.5-25.5)
        size_choice = rng.choices(["Small", "Medium", "Large"], weights=[0.25, 0.55, 0.20], k=1)[0]
        if size_choice == "Small":
            diameter_mm = rng.uniform(15.0, 17.8)
        elif size_choice == "Medium":
            diameter_mm = rng.uniform(18.2, 21.8)
        else:
            diameter_mm = rng.uniform(22.2, 25.5)

        radius_px = int(round((diameter_mm * px_per_mm) / 2.0))

        # Find non-overlapping position
        placed = False
        for _ in range(150):
            cx = rng.randint(radius_px + 50, w - radius_px - 50)
            cy = rng.randint(radius_px + 50, h - radius_px - 50)

            # Check distance to other nuts
            too_close = False
            for ox, oy, orad in placed_nuts:
                dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                if dist < (radius_px + orad + 12):
                    too_close = True
                    break
            if not too_close:
                placed_nuts.append((cx, cy, radius_px))
                placed = True
                break

        if not placed:
            continue

        # Render Macadamia Nut
        bbox, label_name = _draw_macadamia_kernel(mat, cx, cy, radius_px, defect_class, np_rng)

        annotations.append({
            "nut_id": i + 1,
            "bbox": bbox,
            "center": (cx, cy),
            "diameter_mm": round(diameter_mm, 2),
            "size_grade": size_choice,
            "defect_class_id": defect_class,
            "defect_label": label_name,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mat)
    logger.info("Generated batch image with %d nuts: %s", len(annotations), output_path)

    return annotations


def _draw_macadamia_kernel(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    defect_class: int,
    rng: np.random.RandomState,
) -> Tuple[List[int], str]:
    """Draws an individual macadamia kernel with realistic shading, shape, and defects."""
    # Macadamia shape: slight organic deviation from perfect circle
    axes = (int(r * rng.uniform(0.92, 1.05)), int(r * rng.uniform(0.90, 1.02)))
    angle = rng.uniform(0, 180)

    # 1. Soft Drop Shadow
    shadow_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(shadow_mask, (cx + 4, cy + 5), (axes[0] + 3, axes[1] + 3), angle, 0, 360, 255, -1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    for c in range(3):
        canvas[:, :, c] = np.clip(
            canvas[:, :, c].astype(np.int16) - (shadow_mask * 0.35).astype(np.int16), 0, 255
        ).astype(np.uint8)

    # 2. Kernel Body (Cream/Ivory color with 3D spherical gradient)
    kernel_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.ellipse(kernel_mask, (cx, cy), axes, angle, 0, 360, 255, -1)

    # Base ivory/cream tone: BGR ~ [175, 220, 240]
    y_coords, x_coords = np.where(kernel_mask > 0)
    dx = (x_coords - (cx - r * 0.25)) / (r * 1.1)
    dy = (y_coords - (cy - r * 0.25)) / (r * 1.1)
    dist_sq = np.clip(dx**2 + dy**2, 0, 1)

    # Spherical highlight & edge shadow
    shading = (1.0 - 0.45 * dist_sq)[:, np.newaxis]
    base_color = np.array([170, 218, 238], dtype=np.float32)  # BGR
    shaded_colors = np.clip(base_color * shading, 0, 255).astype(np.uint8)

    canvas[y_coords, x_coords] = shaded_colors

    # 3. Add Defect Characteristics
    label_name = "Healthy / Premium"

    if defect_class == 1:
        # Cracked / Broken: Sharp jagged dark fissure across the nut
        label_name = "Cracked / Broken"
        fx1 = int(cx - r * rng.uniform(0.5, 0.8))
        fy1 = int(cy + r * rng.uniform(-0.5, 0.5))
        fx2 = int(cx + r * rng.uniform(0.5, 0.8))
        fy2 = int(cy + r * rng.uniform(-0.5, 0.5))

        # Draw fissure line
        pts = [(fx1, fy1)]
        cur_x, cur_y = fx1, fy1
        steps = 5
        for s in range(1, steps + 1):
            target_x = int(fx1 + (fx2 - fx1) * (s / steps))
            target_y = int(fy1 + (fy2 - fy1) * (s / steps) + rng.uniform(-6, 6))
            cv2.line(canvas, (cur_x, cur_y), (target_x, target_y), (40, 50, 70), 2)
            cur_x, cur_y = target_x, target_y

    elif defect_class == 2:
        # Discolored / Brown Center: Internal enzymatic browning patch
        label_name = "Discolored / Brown Center"
        patch_r = int(r * rng.uniform(0.40, 0.65))
        patch_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.circle(patch_mask, (cx, cy), patch_r, 255, -1)
        patch_mask = cv2.bitwise_and(patch_mask, kernel_mask)
        patch_mask = cv2.GaussianBlur(patch_mask, (9, 9), 0)

        # Blend dark brown color: BGR [45, 80, 130]
        brown_color = np.array([45, 80, 130], dtype=np.float32)
        p_y, p_x = np.where(patch_mask > 0)
        alpha = (patch_mask[p_y, p_x] / 255.0 * 0.75)[:, np.newaxis]
        canvas[p_y, p_x] = np.clip(
            (1.0 - alpha) * canvas[p_y, p_x] + alpha * brown_color, 0, 255
        ).astype(np.uint8)

    elif defect_class == 3:
        # Moldy / Insect Damaged: Dark bore hole with fuzzy greenish mycelium halo
        label_name = "Moldy / Insect Damaged"
        hx = int(cx + rng.uniform(-r * 0.35, r * 0.35))
        hy = int(cy + rng.uniform(-r * 0.35, r * 0.35))

        # Greenish-gray mold halo
        halo_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.circle(halo_mask, (hx, hy), int(r * 0.45), 255, -1)
        halo_mask = cv2.bitwise_and(halo_mask, kernel_mask)
        halo_mask = cv2.GaussianBlur(halo_mask, (11, 11), 0)

        mold_color = np.array([70, 105, 90], dtype=np.float32)  # Greenish-gray BGR
        h_y, h_x = np.where(halo_mask > 0)
        alpha = (halo_mask[h_y, h_x] / 255.0 * 0.65)[:, np.newaxis]
        canvas[h_y, h_x] = np.clip(
            (1.0 - alpha) * canvas[h_y, h_x] + alpha * mold_color, 0, 255
        ).astype(np.uint8)

        # Central necrotic borehole
        cv2.circle(canvas, (hx, hy), int(r * 0.16), (20, 25, 30), -1)

    # Bounding box
    x1 = max(0, cx - axes[0] - 2)
    y1 = max(0, cy - axes[1] - 2)
    x2 = min(canvas.shape[1], cx + axes[0] + 2)
    y2 = min(canvas.shape[0], cy + axes[1] + 2)

    return [x1, y1, x2, y2], label_name


def generate_coco_dataset(
    output_dir: str,
    num_batches: int = 4,
):
    """Generates a complete dataset with raw photos, annotations, and segmented crops."""
    raw_dir = os.path.join(output_dir, "raw")
    annot_dir = os.path.join(output_dir, "annotations")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)

    coco_data = {
        "info": {"description": "Macadamia Nut Quality Grading Benchmark Dataset", "year": 2026},
        "categories": [
            {"id": 0, "name": "Healthy / Premium"},
            {"id": 1, "name": "Cracked / Broken"},
            {"id": 2, "name": "Discolored / Brown Center"},
            {"id": 3, "name": "Moldy / Insect Damaged"},
        ],
        "images": [],
        "annotations": [],
    }

    annot_id = 1

    batches_spec = [
        ("batch_01_premium.jpg", 16, "blue", (0.88, 0.06, 0.06, 0.00), 101),
        ("batch_02_commercial.jpg", 18, "blue", (0.65, 0.18, 0.12, 0.05), 102),
        ("batch_03_high_defect.jpg", 15, "green", (0.45, 0.20, 0.15, 0.20), 103),
        ("batch_04_mixed_sizes.jpg", 20, "blue", (0.75, 0.10, 0.10, 0.05), 104),
    ]

    for img_id, (fname, count, mat_col, ratios, seed) in enumerate(batches_spec, start=1):
        img_path = os.path.join(raw_dir, fname)
        nuts_annot = generate_synthetic_batch(
            output_path=img_path,
            num_nuts=count,
            mat_color=mat_col,
            defect_ratios=ratios,
            seed=seed,
        )

        coco_data["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": 800,
            "height": 800,
        })

        for n in nuts_annot:
            x1, y1, x2, y2 = n["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            coco_data["annotations"].append({
                "id": annot_id,
                "image_id": img_id,
                "category_id": n["defect_class_id"],
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "attributes": {
                    "diameter_mm": n["diameter_mm"],
                    "size_grade": n["size_grade"],
                    "defect_label": n["defect_label"],
                },
            })
            annot_id += 1

    coco_json_path = os.path.join(annot_dir, "coco_annotations.json")
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    logger.info("Saved COCO format annotations to: %s", coco_json_path)


if __name__ == "__main__":
    base_data_dir = os.path.join(os.path.dirname(__file__))
    generate_coco_dataset(base_data_dir)
    print("Dataset generation complete!")
