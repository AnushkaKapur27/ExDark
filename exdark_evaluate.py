#!/usr/bin/env python3
"""
Evaluate a YOLO (Ultralytics) detector on each of the 10 ExDark lighting conditions.

Requires the dataset produced by exdark_prepare.py.
Outputs benchmark_results.csv and prints a table (mAP per condition, macro average,
pooled test set, and correlation between mean luminance and mAP).

Example:
  python exdark_evaluate.py --data_root /content/exdark_yolo --weights runs/detect/train/weights/best.pt
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

CONDITIONS = ["Low", "Ambient", "Object", "Single", "Weak",
              "Strong", "Screen", "Window", "Shadow", "Twilight"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf for proper mAP")
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    root = Path(args.data_root)
    summary = json.loads((root / "benchmark_summary.json").read_text())
    model = YOLO(args.weights)

    def run(yaml_name, tag):
        m = model.val(data=str(root / yaml_name), split="val", imgsz=args.imgsz,
                      batch=args.batch, device=args.device, conf=args.conf, iou=args.iou,
                      project=str(root / "eval_runs"), name=tag, exist_ok=True,
                      plots=False, verbose=False)
        b = m.box
        return dict(precision=float(b.mp), recall=float(b.mr),
                    map50=float(b.map50), map75=float(b.map75), map50_95=float(b.map))

    results = []
    for c in CONDITIONS:
        info = summary["conditions"][c]
        if info["test"] == 0:
            print(f"[{c}] no test images, skipped")
            continue
        print(f"[{c}] evaluating {info['test']} images ...")
        r = run(f"exdark_test_{c.lower()}.yaml", c.lower())
        r.update(condition=c, images=info["test"], instances=info["test_instances"],
                 mean_luminance=info["test_mean_luminance"])
        results.append(r)

    print("[ALL] evaluating pooled test set ...")
    pooled = run("exdark_test_all.yaml", "all")

    cols = ["condition", "images", "instances", "mean_luminance",
            "precision", "recall", "map50", "map75", "map50_95"]
    with open(root / "benchmark_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in results:
            wr.writerow({k: r[k] for k in cols})
        macro = {k: float(np.mean([r[k] for r in results])) for k in
                 ["precision", "recall", "map50", "map75", "map50_95"]}
        wr.writerow(dict(condition="MACRO_AVG", **macro))
        wr.writerow(dict(condition="POOLED", **pooled))

    print(f"\n{'condition':<10}{'imgs':>6}{'meanL':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for r in results:
        print(f"{r['condition']:<10}{r['images']:>6}{r['mean_luminance']:>8.1f}"
              f"{r['map50']:>8.3f}{r['map50_95']:>10.3f}")
    print(f"{'MACRO':<10}{'':>6}{'':>8}{macro['map50']:>8.3f}{macro['map50_95']:>10.3f}")
    print(f"{'POOLED':<10}{'':>6}{'':>8}{pooled['map50']:>8.3f}{pooled['map50_95']:>10.3f}")

    if len(results) >= 3:
        corr = np.corrcoef([r["mean_luminance"] for r in results],
                           [r["map50_95"] for r in results])[0, 1]
        print(f"\nPearson r (mean luminance vs mAP50-95 across conditions): {corr:.3f}")
    print(f"Saved {root / 'benchmark_results.csv'}")


if __name__ == "__main__":
    main()
