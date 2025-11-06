#!/usr/bin/env python
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


DATASETS = {
    "breast_nucls": {"path": "outputs/breast_nucls_train_sample.jsonl", "group": "nucleus"},
    "conic": {"path": "outputs/conic_train.jsonl", "group": "nucleus"},
    "lizard": {"path": "outputs/lizard_train.jsonl", "group": "nucleus"},
    "midog21": {"path": "outputs/midog21_train_sample.jsonl", "group": "nucleus"},
    "midog22": {"path": "outputs/midog22_train.jsonl", "group": "nucleus"},
    "monusac": {"path": "outputs/monusac_train.jsonl", "group": "nucleus"},
    "nuclick": {"path": "outputs/nuclick_train.jsonl", "group": "nucleus"},
    "segpc": {"path": "outputs/segpc_train.jsonl", "group": "cell"},
    "panuke": {"path": "outputs/panuke_train.jsonl", "group": "cell"},
    "puma": {"path": "outputs/puma_train.jsonl", "group": "nucleus"},
}


def _quantiles(array, qs):
    if not array:
        return {q: math.nan for q in qs}
    array = sorted(array)
    n = len(array)
    out = {}
    for q in qs:
        idx = q * (n - 1)
        low = int(math.floor(idx))
        high = int(math.ceil(idx))
        if low == high:
            out[q] = array[low]
        else:
            frac = idx - low
            out[q] = array[low] * (1 - frac) + array[high] * frac
    return out


def compute_stats(path):
    widths, heights, areas, rel_areas = [], [], [], []
    count = 0
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            img_w = rec.get("width", 0) or 0
            img_h = rec.get("height", 0) or 0
            img_area = img_w * img_h if img_w and img_h else None
            for region in rec.get("grounding", {}).get("regions", []):
                x0, y0, x1, y1 = region["bbox"]
                w = max(0.0, float(x1) - float(x0))
                h = max(0.0, float(y1) - float(y0))
                if w <= 0 or h <= 0:
                    continue
                area = w * h
                widths.append(w)
                heights.append(h)
                areas.append(area)
                if img_area:
                    rel_areas.append(area / img_area)
                count += 1
    return {
        "boxes": count,
        "width": {"mean": float(sum(widths) / len(widths)) if widths else math.nan,
                  "median": _quantiles(widths, [0.5])[0.5] if widths else math.nan},
        "height": {"mean": float(sum(heights) / len(heights)) if heights else math.nan,
                   "median": _quantiles(heights, [0.5])[0.5] if heights else math.nan},
        "area": {"mean": float(sum(areas) / len(areas)) if areas else math.nan,
                 "median": _quantiles(areas, [0.5])[0.5] if areas else math.nan},
        "relative_area": {
            "mean": float(sum(rel_areas) / len(rel_areas)) if rel_areas else math.nan,
            "median": _quantiles(rel_areas, [0.5])[0.5] if rel_areas else math.nan,
            "p95": _quantiles(rel_areas, [0.95])[0.95] if rel_areas else math.nan,
        },
    }


def aggregate_group(stats, group_name):
    widths, heights, areas, rel_areas = [], [], [], []
    total_boxes = 0
    for ds, ds_stats in stats.items():
        if DATASETS[ds]["group"] != group_name:
            continue
        total_boxes += ds_stats["boxes"]
        widths.append((ds_stats["width"]["mean"], ds_stats["boxes"]))
        heights.append((ds_stats["height"]["mean"], ds_stats["boxes"]))
        areas.append((ds_stats["area"]["mean"], ds_stats["boxes"]))
        rel_areas.append((ds_stats["relative_area"]["mean"], ds_stats["boxes"]))

    def weighted_mean(values):
        num = sum(v * w for v, w in values if not math.isnan(v))
        den = sum(w for v, w in values if not math.isnan(v))
        return num / den if den else math.nan

    return {
        "boxes": total_boxes,
        "width_mean": weighted_mean(widths),
        "height_mean": weighted_mean(heights),
        "area_mean": weighted_mean(areas),
        "relative_area_mean": weighted_mean(rel_areas),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute bbox statistics for ODVG datasets.")
    parser.add_argument("--output", default="docs/datasets/path_sam_bbox_stats.json")
    args = parser.parse_args()

    per_dataset = {}
    for name, cfg in DATASETS.items():
        stats = compute_stats(cfg["path"])
        stats["group"] = cfg["group"]
        per_dataset[name] = stats

    per_group = {g: aggregate_group(per_dataset, g) for g in {"cell", "nucleus"}}
    summary = {"datasets": per_dataset, "groups": per_group}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("=== Per-group means ===")
    for group, info in per_group.items():
        print(
            f"{group}: mean width={info['width_mean']:.2f}, mean height={info['height_mean']:.2f}, "
            f"mean area={info['area_mean']:.2f}, mean rel_area={info['relative_area_mean']:.4f} "
            f"(boxes={info['boxes']})"
        )
    print("\n=== Detailed per-dataset stats ===")
    for name, stats in per_dataset.items():
        print(
            f"{name} ({stats['group']}): boxes={stats['boxes']}, "
            f"w_mean={stats['width']['mean']:.2f}, h_mean={stats['height']['mean']:.2f}, "
            f"area_mean={stats['area']['mean']:.2f}, rel_area_mean={stats['relative_area']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
