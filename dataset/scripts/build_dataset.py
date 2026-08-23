#!/usr/bin/env python3
"""Build a simple person/vehicle dataset from COCO.

Example:
    python dataset/scripts/build_dataset.py \
        --annotation /path/to/instances_train2017.json \
        --output-dir /path/to/person_vehicle \
        --train 6000 --val 1000 --test 1000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

PERSON_VEHICLE_CATEGORIES = {1, 2, 3, 4, 5, 7}


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_split_ids(annotations, train_n: int, val_n: int, test_n: int):
    image_ids = sorted({int(a["image_id"]) for a in annotations if int(a.get("category_id", -1)) in PERSON_VEHICLE_CATEGORIES})
    total = train_n + val_n + test_n
    if len(image_ids) < total:
        raise ValueError(f"Requested {total} images but only {len(image_ids)} matching COCO images are available.")

    train_ids = set(image_ids[:train_n])
    val_ids = set(image_ids[train_n : train_n + val_n])
    test_ids = set(image_ids[train_n + val_n : train_n + val_n + test_n])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_split_jsons(annotation_path: str | Path, output_dir: str | Path, train_n: int, val_n: int, test_n: int):
    data = load_json(annotation_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    category_map = {int(cat["id"]): cat for cat in data.get("categories", [])}
    splits = get_split_ids(annotations, train_n, val_n, test_n)

    split_files = {}
    for split_name, ids in splits.items():
        split_images = [img for img in images if int(img["id"]) in ids]
        split_annotations = [
            ann for ann in annotations
            if int(ann.get("image_id", -1)) in ids and int(ann.get("category_id", -1)) in PERSON_VEHICLE_CATEGORIES
        ]
        payload = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": [category_map[i] for i in sorted(PERSON_VEHICLE_CATEGORIES)],
        }
        if "info" in data:
            payload["info"] = data["info"]
        if "licenses" in data:
            payload["licenses"] = data["licenses"]
        out_path = Path(output_dir) / "splits" / f"{split_name}.json"
        save_json(out_path, payload)
        split_files[split_name] = out_path

    print(f"Created split JSONs in {Path(output_dir) / 'splits'}")
    return split_files


def download_split_images(json_path: str | Path, output_dir: str | Path, split_name: str, workers: int = 4):
    data = load_json(json_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    images = data.get("images", [])
    if not images:
        print(f"No images to download for {split_name}")
        return

    def download_one(image):
        file_name = image["file_name"]
        output_file = target_dir / file_name
        if output_file.exists():
            return
        image_url = f"http://images.cocodataset.org/{split_name}/" + file_name
        urlretrieve(image_url, str(output_file))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(download_one, images), total=len(images), desc=f"Downloading {split_name}", unit="img"))

    print(f"Downloaded {len(images)} images to {target_dir}")


def convert_to_yolo(json_path: str | Path, image_dir: str | Path, label_dir: str | Path):
    data = load_json(json_path)
    image_lookup = {int(img["id"]): img for img in data.get("images", [])}
    labels_by_file = {}

    for ann in tqdm(data.get("annotations", []), desc=f"Converting {Path(json_path).stem}", unit="ann"):
        image_id = int(ann.get("image_id", -1))
        image = image_lookup.get(image_id)
        if image is None:
            continue

        width = float(image.get("width", 1))
        height = float(image.get("height", 1))
        x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        if w <= 0 or h <= 0:
            continue

        class_id = 0 if int(ann.get("category_id", -1)) == 1 else 1
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        bw = w / width
        bh = h / height

        stem = Path(image["file_name"]).stem
        labels_by_file.setdefault(stem, []).append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    for stem, lines in labels_by_file.items():
        with open(label_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    print(f"Wrote YOLO labels to {label_dir}")


def build_dataset(annotation_path: str | Path, output_dir: str | Path, train: int, val: int, test: int, workers: int = 4):
    root = Path(output_dir)
    images_root = root / "images"
    labels_root = root / "labels"
    splits_root = root / "splits"

    for split in ("train", "val", "test"):
        shutil.rmtree(images_root / split, ignore_errors=True)
        shutil.rmtree(labels_root / split, ignore_errors=True)
    shutil.rmtree(splits_root, ignore_errors=True)

    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    split_files = build_split_jsons(annotation_path, root, train, val, test)

    for split_name in ("train", "val", "test"):
        split_json = split_files[split_name]
        image_dir = images_root / split_name
        label_dir = labels_root / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{split_name.upper()}] Downloading images...")
        download_split_images(split_json, image_dir, f"{split_name}2017", workers=workers)

        print(f"\n[{split_name.upper()}] Converting to YOLO labels...")
        convert_to_yolo(split_json, image_dir, label_dir)

    print(f"\nDataset built at: {root}")


def main():
    parser = argparse.ArgumentParser(description="Build a simple person/vehicle COCO dataset.")
    parser.add_argument("--annotation", required=True, help="COCO annotation JSON, e.g. instances_train2017.json")
    parser.add_argument("--output-dir", required=True, help="Directory for the final dataset")
    parser.add_argument("--train", type=int, default=3000, help="Train image count")
    parser.add_argument("--val", type=int, default=500, help="Val image count")
    parser.add_argument("--test", type=int, default=500, help="Test image count")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2), help="Number of workers for parallel downloads")
    args = parser.parse_args()

    build_dataset(args.annotation, args.output_dir, args.train, args.val, args.test, workers=args.workers)


if __name__ == "__main__":
    main()
