#!/usr/bin/env python3
"""
ExDark -> YOLO dataset preparation with an illumination-stratified benchmark.

Inputs (from https://github.com/cs-chan/Exclusively-Dark-Image-Dataset):
  ExDark.zip        -> <images_root>/<Class>/<name>.jpg
  ExDark_Annno.zip  -> <annno_root>/<Class>/<name>.jpg.txt   (bbGt format)
  imageclasslist.txt  columns: Name | Class(1-12) | Light(1-10) | In/Out(1,2) | Split(1,2,3)

Outputs (under --output_dir):
  images/{train,val}/                 labels/{train,val}/
  images/test/<condition>/            labels/test/<condition>/     (10 conditions)
  exdark.yaml                         main config (train / val / test=all test images)
  exdark_test_<condition>.yaml        one config per lighting condition (10 files)
  exdark_test_all.yaml                all test images pooled
  manifest.csv                        one row per image (split, light, luminance, #boxes ...)
  benchmark_summary.json              counts, instances and luminance stats per condition x split
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ExDark class order (alphabetical, matches imageclasslist.txt class ids 1..12)
CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
           "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
CLASS_TO_ID = {c.lower(): i for i, c in enumerate(CLASSES)}

# ExDark lighting ids 1..10 as defined in imageclasslist.txt
CONDITIONS = ["Low", "Ambient", "Object", "Single", "Weak",
              "Strong", "Screen", "Window", "Shadow", "Twilight"]
SPLIT_NAMES = {1: "train", 2: "val", 3: "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_TEST_IMAGES_WARN = 30


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def read_image_class_list(path):
    """Return a list of dicts: name, class_id, light_id, indoor_outdoor, split_id."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            try:
                cls, light, io, split = (int(x) for x in p[-4:])
            except ValueError:
                continue  # header line
            name = " ".join(p[:-4])
            if not (1 <= cls <= 12 and 1 <= light <= 10 and split in SPLIT_NAMES):
                continue
            rows.append(dict(name=name, class_id=cls - 1, light_id=light - 1,
                             indoor_outdoor=io, split_id=split))
    return rows


def build_index(root, exts=None, strip_txt=False):
    """Index files under root by lowercase full name and by lowercase stem."""
    full, stem = {}, {}
    dupes = 0
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        if strip_txt:
            if p.suffix.lower() != ".txt":
                continue
            key = p.name[:-4].lower()          # "2015_00001.jpg.txt" -> "2015_00001.jpg"
        else:
            key = p.name.lower()
        if key in full:
            dupes += 1
        full[key] = p
        stem.setdefault(Path(key).stem if Path(key).suffix.lower() in IMG_EXTS else key, p)
    if dupes:
        print(f"  warning: {dupes} duplicate file names under {root}")
    return full, stem


def lookup(index, name):
    full, stem = index
    key = name.lower()
    if key in full:
        return full[key]
    return stem.get(Path(key).stem)


def parse_annotation(path, w, h):
    """
    ExDark bbGt line: <Class> l t w h <flags...>   (pixels, top-left origin).
    Returns (yolo_lines, Counter of issues).
    """
    issues = Counter()
    out = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line[0] in "%#":
            continue
        p = line.split()
        if len(p) < 5:
            issues["malformed"] += 1
            continue
        cid = CLASS_TO_ID.get(p[0].lower())
        if cid is None:
            issues["unknown_class"] += 1
            continue
        try:
            l, t, bw, bh = (float(x) for x in p[1:5])
        except ValueError:
            issues["malformed"] += 1
            continue
        x1, y1 = max(0.0, l), max(0.0, t)
        x2, y2 = min(float(w), l + bw), min(float(h), t + bh)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            issues["degenerate"] += 1
            continue
        if (x1, y1, x2, y2) != (l, t, l + bw, t + bh):
            issues["clipped"] += 1
        out.append(f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                   f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out, issues


def mean_luminance(img_rgb):
    """Mean of L = 0.299R + 0.587G + 0.114B over all pixels (0-255)."""
    a = np.asarray(img_rgb, dtype=np.float32)
    return float(np.mean(0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]))


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def assign_random_split(records, train_ratio, val_ratio, seed):
    """Seeded split, stratified by lighting condition (each condition split separately)."""
    rng = random.Random(seed)
    by_cond = defaultdict(list)
    for r in records:
        by_cond[r["light_id"]].append(r)
    for light_id in sorted(by_cond):
        group = sorted(by_cond[light_id], key=lambda r: r["name"])
        rng.shuffle(group)
        n_train = int(round(len(group) * train_ratio))
        n_val = int(round(len(group) * val_ratio))
        for i, r in enumerate(group):
            r["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"


# --------------------------------------------------------------------------- #
# YAML
# --------------------------------------------------------------------------- #
def write_yaml(path, root, train, val, test):
    lines = [f"path: {root}", f"train: {train}", f"val: {val}", f"test: {test}", "", "names:"]
    lines += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_root", required=True, help="Folder from ExDark.zip (contains class sub-folders)")
    ap.add_argument("--annno_root", required=True, help="Folder from ExDark_Annno.zip")
    ap.add_argument("--imageclasslist", required=True, help="Path to imageclasslist.txt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", choices=["official", "random"], default="official",
                    help="official = train/val/test column of imageclasslist.txt (default); "
                         "random = seeded split stratified by lighting condition")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="random split only")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="random split only")
    ap.add_argument("--seed", type=int, default=0, help="random split only")
    ap.add_argument("--overwrite", action="store_true", help="delete output_dir first if it exists")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            sys.exit(f"{out} is not empty. Use --overwrite to replace it.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_image_class_list(args.imageclasslist)
    if not rows:
        sys.exit("No valid rows found in imageclasslist.txt")
    print(f"imageclasslist: {len(rows)} entries")

    print("Indexing images and annotations ...")
    img_index = build_index(args.images_root, IMG_EXTS)
    ann_index = build_index(args.annno_root, strip_txt=True)
    print(f"  {len(img_index[0])} images, {len(ann_index[0])} annotation files found")

    # ---- Pass 1: validate, convert labels, measure luminance ----------------
    records, skipped = [], Counter()
    issue_totals = Counter()
    exif_rotated = 0
    for i, r in enumerate(rows, 1):
        img_path = lookup(img_index, r["name"])
        if img_path is None:
            skipped["image_not_found"] += 1
            continue
        ann_path = lookup(ann_index, r["name"])
        if ann_path is None:
            skipped["annotation_not_found"] += 1
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                if im.getexif().get(274, 1) not in (1, None):
                    exif_rotated += 1
                lum = mean_luminance(im.convert("RGB"))
        except Exception:
            skipped["image_unreadable"] += 1
            continue
        labels, issues = parse_annotation(ann_path, w, h)
        issue_totals.update(issues)
        if not labels:
            skipped["no_valid_boxes"] += 1
            continue
        r.update(img_path=img_path, width=w, height=h, luminance=lum, labels=labels,
                 condition=CONDITIONS[r["light_id"]], split=SPLIT_NAMES[r["split_id"]])
        records.append(r)
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(rows)}")

    if not records:
        sys.exit("No usable images. Check the three input paths.")

    if args.split == "random":
        assign_random_split(records, args.train_ratio, args.val_ratio, args.seed)

    # ---- Pass 2: write dataset ------------------------------------------------
    for r in records:
        sub = r["split"] if r["split"] != "test" else f"test/{r['condition'].lower()}"
        img_dir = out / "images" / sub
        lbl_dir = out / "labels" / sub
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dest_name = r["img_path"].name
        shutil.copy2(r["img_path"], img_dir / dest_name)
        (lbl_dir / f"{Path(dest_name).stem}.txt").write_text("\n".join(r["labels"]) + "\n")
        r["dest"] = f"images/{sub}/{dest_name}"

    # Make sure all 10 test folders exist even if a condition has no test images
    for c in CONDITIONS:
        (out / "images" / "test" / c.lower()).mkdir(parents=True, exist_ok=True)
        (out / "labels" / "test" / c.lower()).mkdir(parents=True, exist_ok=True)

    # ---- YAML configs ---------------------------------------------------------
    root = str(out.resolve())
    write_yaml(out / "exdark.yaml", root, "images/train", "images/val", "images/test")
    write_yaml(out / "exdark_test_all.yaml", root, "images/train", "images/test", "images/test")
    for c in CONDITIONS:
        d = f"images/test/{c.lower()}"
        write_yaml(out / f"exdark_test_{c.lower()}.yaml", root, "images/train", d, d)

    # ---- Manifest -------------------------------------------------------------
    with open(out / "manifest.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "class", "condition", "indoor_outdoor", "split",
                     "width", "height", "mean_luminance", "num_boxes"])
        for r in sorted(records, key=lambda x: x["name"]):
            wr.writerow([r["dest"], CLASSES[r["class_id"]], r["condition"],
                         "indoor" if r["indoor_outdoor"] == 1 else "outdoor", r["split"],
                         r["width"], r["height"], f"{r['luminance']:.3f}", len(r["labels"])])

    # ---- Summary --------------------------------------------------------------
    summary = {"split_mode": args.split, "num_images": len(records),
               "skipped": dict(skipped), "annotation_issues": dict(issue_totals),
               "conditions": {}}
    print("\n=== Illumination-stratified benchmark ===")
    print(f"{'condition':<9} {'train':>6} {'val':>5} {'test':>5} {'test boxes':>11} "
          f"{'mean L':>7} {'std L':>6}")
    for c in CONDITIONS:
        recs = [r for r in records if r["condition"] == c]
        counts = Counter(r["split"] for r in recs)
        test = [r for r in recs if r["split"] == "test"]
        lums = [r["luminance"] for r in recs]
        test_boxes = sum(len(r["labels"]) for r in test)
        per_class = Counter()
        for r in test:
            for ln in r["labels"]:
                per_class[CLASSES[int(ln.split()[0])]] += 1
        summary["conditions"][c] = dict(
            train=counts["train"], val=counts["val"], test=counts["test"],
            test_instances=test_boxes, test_instances_per_class=dict(per_class),
            mean_luminance=float(np.mean(lums)) if lums else None,
            std_luminance=float(np.std(lums)) if lums else None,
            test_mean_luminance=float(np.mean([r["luminance"] for r in test])) if test else None)
        print(f"{c:<9} {counts['train']:>6} {counts['val']:>5} {counts['test']:>5} {test_boxes:>11} "
              f"{np.mean(lums) if lums else float('nan'):>7.1f} {np.std(lums) if lums else float('nan'):>6.1f}")
        if counts["test"] < MIN_TEST_IMAGES_WARN:
            print(f"  warning: only {counts['test']} test images for '{c}'; "
                  f"per-condition metrics will be noisy")
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nUsable images: {len(records)} / {len(rows)}")
    if skipped:
        print("Skipped:", dict(skipped))
    if issue_totals:
        print("Annotation issues (box level):", dict(issue_totals))
    if exif_rotated:
        print(f"Note: {exif_rotated} images carry an EXIF rotation tag. Boxes are used as-is "
              f"relative to the stored pixel grid; spot-check a few if results look off.")
    print(f"\nDone. Dataset: {root}")


if __name__ == "__main__":
    main()
