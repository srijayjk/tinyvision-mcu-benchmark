#!/usr/bin/env python3
"""Visualize a few COCO images with bounding boxes.

Example:
    python dataset/scripts/visualize_annotations.py \
        --annotation dataset/person_vehicle/splits/val.json \
        --images-dir dataset/person_vehicle/images/val \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
    3: "tab:red",
    4: "tab:purple",
    5: "tab:brown",
    6: "tab:pink",
    7: "tab:gray",
    8: "tab:olive",
    9: "tab:cyan",
}


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_box(ax, bbox, label, color):
    x, y, w, h = [float(v) for v in bbox]
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(x, max(0, y - 6), label, color=color, fontsize=9, fontweight="bold", va="bottom")


def visualize_annotations(annotation_path: str | Path, images_dir: str | Path, limit: int = 5, save_dir: str | Path | None = None):
    data = load_json(annotation_path)
    image_lookup = {int(img["id"]): img for img in data.get("images", [])}
    category_lookup = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}

    # Keep a balanced sample: 5 person images and 5 vehicle images.
    category_to_class = {}
    for category_id in category_lookup:
        category_name = category_lookup[category_id].lower()
        if category_name == "person":
            category_to_class[category_id] = 0
        elif category_name in {"bicycle", "car", "motorcycle", "bus", "truck", "van", "train"}:
            category_to_class[category_id] = 1

    annotations_by_image = {}
    class_image_ids = {0: [], 1: []}
    for ann in data.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        category_id = int(ann.get("category_id", -1))
        class_id = category_to_class.get(category_id)
        if class_id is None:
            continue

        annotations_by_image.setdefault(image_id, []).append(ann)
        class_image_ids[class_id].append(image_id)

    selected_ids = []
    for class_id in (0, 1):
        unique_ids = sorted(set(class_image_ids.get(class_id, [])))[:limit]
        selected_ids.extend(unique_ids)
    selected_ids = list(dict.fromkeys(selected_ids))

    if not selected_ids:
        raise ValueError(f"No person or vehicle images found in {annotation_path}.")

    valid_ids = []
    for image_id in selected_ids:
        image_info = image_lookup.get(image_id)
        if image_info is None:
            continue
        file_name = image_info["file_name"]
        image_path = Path(images_dir) / file_name
        if image_path.exists():
            valid_ids.append(image_id)
        else:
            print(f"Skipping missing file: {image_path}")

    if not valid_ids:
        raise ValueError(f"No valid image files were found in {images_dir} for the selected annotations.")

    fig, axes = plt.subplots(len(valid_ids), 1, figsize=(12, 5 * len(valid_ids)))
    if len(valid_ids) == 1:
        axes = [axes]

    for ax, image_id in zip(axes, valid_ids):
        image_info = image_lookup[image_id]
        file_name = image_info["file_name"]
        image_path = Path(images_dir) / file_name
        image = plt.imread(str(image_path))
        ax.imshow(image)
        ax.set_title(f"{file_name} | id={image_id}")
        ax.axis("off")

        for ann in annotations_by_image.get(image_id, []):
            category_id = int(ann.get("category_id", -1))
            class_id = category_to_class.get(category_id)
            if class_id is None:
                continue
            category_name = category_lookup.get(category_id, str(category_id))
            bbox = ann.get("bbox", [0, 0, 0, 0])
            color = DEFAULT_COLORS.get(class_id, "tab:blue")
            draw_box(ax, bbox, category_name, color)

    plt.tight_layout()

    if save_dir is not None:
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, image_id in enumerate(valid_ids):
            axes[idx].figure.savefig(out_dir / f"sample_{image_id}.png", bbox_inches="tight")
        print(f"Saved {len(valid_ids)} image plots to {out_dir}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO sample images with bounding boxes.")
    parser.add_argument("--annotation", required=True, help="Path to a COCO JSON file, such as val.json or a subset JSON")
    parser.add_argument("--images-dir", required=True, help="Directory containing the corresponding images")
    parser.add_argument("--limit", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--save-dir", default=None, help="Optional folder to save sample previews as PNG files")
    args = parser.parse_args()

    visualize_annotations(args.annotation, args.images_dir, limit=args.limit, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
