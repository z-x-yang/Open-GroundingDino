import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy import ndimage
except Exception:
    ndimage = None

BBox = Tuple[int, int, int, int]


def _safe_open_image(path: str) -> Tuple[Image.Image, int, int]:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    return img, width, height


def _clamp_bbox(bbox: Sequence[int], width: int, height: int) -> BBox:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width - 1))
    x1 = max(0, min(int(x1), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    y1 = max(0, min(int(y1), height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _iter_unique_values(arr: np.ndarray) -> Iterable[int]:
    vals = np.unique(arr)
    for v in vals:
        if v > 0:
            yield int(v)


def _mask_to_bboxes(mask: np.ndarray, min_size: int = 4) -> List[BBox]:
    """
    Convert an integer mask where each unique positive value corresponds to
    an instance into bounding boxes (x0,y0,x1,y1). The mask can be 2-D or 3-D;
    if 3-D the last dimension is treated as independent masks per class channel.
    """
    if mask.ndim == 3:
        boxes = []
        for ch in range(mask.shape[-1]):
            boxes.extend(_mask_to_bboxes(mask[..., ch], min_size))
        return boxes

    boxes: List[BBox] = []
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return boxes
    inst_ids = np.unique(mask[ys, xs])
    inst_ids = inst_ids[inst_ids > 0]
    for inst_id in inst_ids:
        ys_, xs_ = np.where(mask == inst_id)
        if ys_.size == 0:
            continue
        y0 = int(ys_.min())
        y1 = int(ys_.max()) + 1
        x0 = int(xs_.min())
        x1 = int(xs_.max()) + 1
        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            continue
        boxes.append((x0, y0, x1, y1))
    return boxes


def read_ihc_tlymphoctype(root_dir: str) -> List[Dict]:
    """
    Read ROI-level CSV annotations with columns XMin/YMin/XMax/YMax and Cell Type.
    """
    if not os.path.isdir(root_dir):
        return []
    records: List[Dict] = []
    for img_path in Path(root_dir).glob("*.tif"):
        csv_path = img_path.with_suffix(".csv")
        if not csv_path.exists():
            csv_path = Path(str(img_path).replace(".tif", "_consensus.csv"))
        if not csv_path.exists():
            continue
        try:
            img, width, height = _safe_open_image(str(img_path))
        except Exception:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        need = {"xmin", "ymin", "xmax", "ymax"}
        if not need.issubset(cols.keys()):
            continue
        label_col = None
        for cand in ["cell type", "Cell Type", "label", "Label"]:
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            continue
        instances: List[Dict] = []
        for _, row in df.iterrows():
            try:
                bbox = (
                    int(row[cols["xmin"]]),
                    int(row[cols["ymin"]]),
                    int(row[cols["xmax"]]),
                    int(row[cols["ymax"]]),
                )
            except Exception:
                continue
            bbox = _clamp_bbox(bbox, width, height)
            if not bbox:
                continue
            label = str(row[label_col]).strip()
            instances.append({"bbox": bbox, "label": label})
        if instances:
            records.append(
                {
                    "image_path": str(img_path),
                    "width": width,
                    "height": height,
                    "instances": instances,
                }
            )
    return records


def read_midog_json(json_path: str, images_dir: str) -> List[Dict]:
    """Generic reader for MIDOG-style bbox json with image filenames and bboxes."""
    records: Dict[str, Dict] = {}
    if not (os.path.isfile(json_path) and os.path.isdir(images_dir)):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    img_id_to_name = {d["id"]: d["file_name"] for d in data.get("images", [])}
    img_id_to_wh = {
        d["id"]: (d.get("width", 0), d.get("height", 0))
        for d in data.get("images", [])
    }
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        img_name = img_id_to_name.get(img_id)
        if not img_name:
            continue
        if img_name not in records:
            width, height = img_id_to_wh.get(img_id, (0, 0))
            records[img_name] = {
                "image_path": os.path.join(images_dir, img_name),
                "width": width,
                "height": height,
                "instances": [],
            }
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        bbox = (int(x), int(y), int(x + w), int(y + h))
        records[img_name]["instances"].append({"bbox": bbox, "label": "mitoses"})
    return list(records.values())


def read_midog21(json_path: str, images_dir: str) -> List[Dict]:
    return read_midog_json(json_path, images_dir)


def read_breast_nucls(root_dir: str) -> List[Dict]:
    drop_names = {"fov", "unlabled", "unknown"}
    records: Dict[str, Dict] = {}

    def map_case(csv_name: str) -> str:
        caseimg = csv_name.split("#_")[-1]
        head = caseimg.split("_")[0]
        parts = head.split("-")
        if len(parts) == 6:
            tcga = "-".join(parts[:3] + [parts[-1]])
            caseimg = "_".join([tcga] + caseimg.split("_")[1:])
        return caseimg

    for split in ["train", "eval"]:
        rgb_dir = os.path.join(root_dir, split, "rgb")
        csv_dir = os.path.join(root_dir, split, "csv")
        if not (os.path.isdir(rgb_dir) and os.path.isdir(csv_dir)):
            continue
        for csv_path in Path(csv_dir).glob("*.csv"):
            img_name = f"{map_case(csv_path.stem)}.png"
            img_path = Path(rgb_dir) / img_name
            if not img_path.exists():
                continue
            try:
                img, width, height = _safe_open_image(str(img_path))
            except Exception:
                continue
            if str(img_path) not in records:
                records[str(img_path)] = {
                    "image_path": str(img_path),
                    "width": width,
                    "height": height,
                    "instances": [],
                }
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            for _, row in df.iterrows():
                label = (
                    str(
                        row.get(
                            cols.get("main_classification")
                            or cols.get("raw_classification"),
                            "",
                        )
                    )
                    .strip()
                    .lower()
                )
                if not label or label in drop_names:
                    continue
                if "type" in cols:
                    row_type = str(row[cols["type"]]).lower()
                else:
                    row_type = "rectangles"
                if row_type.startswith("rect") and all(
                    key in cols for key in ["xmin", "ymin", "xmax", "ymax"]
                ):
                    try:
                        bbox = (
                            int(row[cols["xmin"]]),
                            int(row[cols["ymin"]]),
                            int(row[cols["xmax"]]),
                            int(row[cols["ymax"]]),
                        )
                    except Exception:
                        continue
                elif all(key in cols for key in ["coords_x", "coords_y"]):
                    try:
                        xs = [
                            int(float(v))
                            for v in str(row[cols["coords_x"]]).split(",")
                            if v != ""
                        ]
                        ys = [
                            int(float(v))
                            for v in str(row[cols["coords_y"]]).split(",")
                            if v != ""
                        ]
                        if not xs or not ys:
                            continue
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                    except Exception:
                        continue
                else:
                    continue
                bbox = _clamp_bbox(bbox, width, height)
                if not bbox:
                    continue
                if (bbox[2] - bbox[0]) < 4 or (bbox[3] - bbox[1]) < 4:
                    continue
                records[str(img_path)]["instances"].append(
                    {"bbox": bbox, "label": label}
                )
    return [rec for rec in records.values() if rec["instances"]]


def read_monusac_processed(root_dir: str) -> List[Dict]:
    """
    Each case folder contains image.png plus per-class *_mask.png bitmaps.
    """
    if not os.path.isdir(root_dir):
        records: List[Dict] = []
    else:
        records = []
        for split in ["Training_image_mask", "Testing_image_mask"]:
            split_dir = Path(root_dir) / split
            if not split_dir.exists():
                continue
            for case_dir in split_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                img_path = case_dir / "image.png"
                if not img_path.exists():
                    continue
                try:
                    img, width, height = _safe_open_image(str(img_path))
                except Exception:
                    continue
                instances: List[Dict] = []
                for mask_path in case_dir.glob("*_mask.png"):
                    label = mask_path.stem.split("_")[0]
                    mask = np.array(Image.open(mask_path))
                    mask = (mask > 0).astype(np.uint8)
                    if ndimage is not None and mask.max() == 1:
                        labeled, num = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
                        if num > 0:
                            boxes = _mask_to_bboxes(labeled, min_size=6)
                        else:
                            boxes = []
                    else:
                        boxes = _mask_to_bboxes(mask, min_size=6)
                    for bbox in boxes:
                        bbox = _clamp_bbox(bbox, width, height)
                        if bbox:
                            instances.append({"bbox": bbox, "label": label})
                if instances:
                    records.append(
                        {
                            "image_path": str(img_path),
                            "width": width,
                            "height": height,
                            "instances": instances,
                        }
                    )
    if records:
        return records

    # fallback: use cropped tiles with npy masks
    fallback_dir = Path(root_dir).parent / "monusac_cropped"
    label_map = {
        1: "Ambiguous",
        2: "Epithelial",
        3: "Macrophage",
        4: "Neutrophil",
        5: "Lymphocyte",
    }
    records: List[Dict] = []
    if fallback_dir.exists():
        for img_path in fallback_dir.glob("*.png"):
            mask_path = img_path.with_suffix(".npy")
            if not mask_path.exists():
                continue
            try:
                img, width, height = _safe_open_image(str(img_path))
            except Exception:
                continue
            try:
                mask = np.load(mask_path)
            except Exception:
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = mask.astype(np.int32)
            instances: List[Dict] = []
            for val, name in label_map.items():
                if val == 1:  # drop ambiguous
                    continue
                channel = (mask == val).astype(np.uint8)
                boxes = _mask_to_bboxes(channel, min_size=4)
                for bbox in boxes:
                    bbox = _clamp_bbox(bbox, width, height)
                    if bbox:
                        instances.append({"bbox": bbox, "label": name})
            if instances:
                records.append(
                    {
                        "image_path": str(img_path),
                        "width": width,
                        "height": height,
                        "instances": instances,
                        "split": split,
                    }
                )
    return records


def read_segpc(root_dir: str) -> List[Dict]:
    """
    SEGPC layout:
      train/train/train/x/*.bmp (images)
      train/train/train/y/*_k.bmp (masks per instance)
    Similar for validation/test.
    """
    NUCLEUS_VALUE = 40
    records: List[Dict] = []
    for split in ["train", "validation"]:
        img_dir = Path(root_dir) / split / split / split / "x"
        mask_dir = Path(root_dir) / split / split / split / "y"
        if not img_dir.exists() or not mask_dir.exists():
            continue
        mask_index = defaultdict(list)
        for mask_path in mask_dir.glob("*.bmp"):
            stem = mask_path.stem
            base = stem.split("_")[0]
            mask_index[base].append(mask_path)
        for img_path in img_dir.glob("*.bmp"):
            masks = mask_index.get(img_path.stem, [])
            if not masks:
                continue
            try:
                img, width, height = _safe_open_image(str(img_path))
            except Exception:
                continue
            instances: List[Dict] = []
            for mask_path in masks:
                mask = np.array(Image.open(mask_path))
                nucleus_mask = (mask == NUCLEUS_VALUE).astype(np.uint8)
                if nucleus_mask.sum() == 0:
                    nucleus_mask = (mask > 0).astype(np.uint8)
                boxes = _mask_to_bboxes(nucleus_mask, min_size=6)
                for bbox in boxes:
                    bbox = _clamp_bbox(bbox, width, height)
                    if bbox:
                        instances.append({"bbox": bbox, "label": "plasma cell"})
            if instances:
                records.append(
                    {
                        "image_path": str(img_path),
                        "width": width,
                        "height": height,
                        "instances": instances,
                    }
                )
    return records


def read_nuclick(root_dir: str) -> List[Dict]:
    records: List[Dict] = []
    splits = [
        ("Train/images", "Train/masks"),
        ("Validation/images", "Validation/masks"),
    ]
    for img_rel, mask_rel in splits:
        img_dir = Path(root_dir) / img_rel
        mask_dir = Path(root_dir) / mask_rel
        if not img_dir.exists() or not mask_dir.exists():
            continue
        for mask_path in mask_dir.glob("*.png"):
            img_path = img_dir / mask_path.name.replace("_mask.png", ".png")
            if not img_path.exists():
                continue
            try:
                img, width, height = _safe_open_image(str(img_path))
            except Exception:
                continue
            mask = np.array(Image.open(mask_path))
            instances: List[Dict] = []
            boxes = _mask_to_bboxes(mask, min_size=6)
            for bbox in boxes:
                bbox = _clamp_bbox(bbox, width, height)
                if bbox:
                    instances.append({"bbox": bbox, "label": "leukocyte"})
            if instances:
                records.append(
                    {
                        "image_path": str(img_path),
                        "width": width,
                        "height": height,
                        "instances": instances,
                    }
                )
    return records


def read_panuke(root_dir: str) -> List[Dict]:
    """
    PanNuke raw layout:
      partX/Images/images.npy (N,256,256,3)
      partX/Masks/masks.npy (N,256,256,6) per dataset release
    We assume PNGs already exist under panuke_processed/p{part_idx}_idx.png;
    if not, fallback to saving temporary PNGs.
    """
    processed_dir = Path(root_dir + "_processed")
    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    class_names = [
        "Neoplastic cells",
        "Inflammatory",
        "Connective/Soft tissue cells",
        "Dead Cells",
        "Epithelial",
        "Background",
    ]
    records: List[Dict] = []

    for part_idx, part_name in enumerate(["part1", "part2", "part3"]):
        img_npy = Path(root_dir) / part_name / "Images" / "images.npy"
        mask_npy = Path(root_dir) / part_name / "Masks" / "masks.npy"
        if not img_npy.exists() or not mask_npy.exists():
            continue
        images = np.load(img_npy, mmap_mode="r")
        masks = np.load(mask_npy, mmap_mode="r")
        count = images.shape[0]
        for idx in range(count):
            img_name = f"p{part_idx}_{idx}.png"
            img_path = processed_dir / img_name
            if not img_path.exists():
                Image.fromarray(images[idx].astype(np.uint8)).save(img_path)
            instances: List[Dict] = []
            for ch, class_name in enumerate(class_names[:-1]):  # drop background
                channel = masks[idx, :, :, ch]
                boxes = _mask_to_bboxes(channel, min_size=4)
                for bbox in boxes:
                    instances.append({"bbox": bbox, "label": class_name})
            if not instances:
                continue
            records.append(
                {
                    "image_path": str(img_path),
                    "width": 256,
                    "height": 256,
                    "instances": instances,
                }
            )
    return records


def read_lizard(root_dir: str) -> List[Dict]:
    label_dir = Path(root_dir) / "lizard_labels" / "Lizard_Labels" / "Labels"
    image_dirs = [
        Path(root_dir) / "lizard_images1" / "Lizard_Images1",
        Path(root_dir) / "lizard_images2" / "Lizard_Images2",
    ]
    records: List[Dict] = []
    import scipy.io as sio
    class_map = {
        1: "Neutrophil",
        2: "Epithelial",
        3: "Lymphocyte",
        4: "Plasma",
        5: "Eosinophil",
        6: "Connective tissue",
    }

    for img_dir in image_dirs:
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*.png"):
            label_path = label_dir / f"{img_path.stem}.mat"
            if not label_path.exists():
                continue
            try:
                img, width, height = _safe_open_image(str(img_path))
            except Exception:
                continue
            try:
                data = sio.loadmat(str(label_path))
            except Exception:
                continue
            classes = data.get("class")
            bboxs = data.get("bbox")
            inst_map = data.get("inst_map")
            if classes is None or bboxs is None or inst_map is None:
                continue
            ids = np.squeeze(data.get("id")).tolist()
            instances: List[Dict] = []
            for inst_id, class_val, bbox in zip(ids, classes, bboxs):
                cls_id = int(class_val[0]) if isinstance(class_val, (list, np.ndarray)) else int(class_val)
                label = class_map.get(cls_id, str(cls_id))
                y1, y2, x1, x2 = bbox
                bbox_xyxy = (int(x1), int(y1), int(x2), int(y2))
                bbox_xyxy = _clamp_bbox(bbox_xyxy, width, height)
                if not bbox_xyxy:
                    continue
                instances.append({"bbox": bbox_xyxy, "label": label})
            if instances:
                records.append(
                    {
                        "image_path": str(img_path),
                        "width": width,
                        "height": height,
                        "instances": instances,
                    }
                )
    return records


def read_conic(root_dir: str) -> List[Dict]:
    data_dir = Path(root_dir) / "data"
    images_path = data_dir / "images.npy"
    labels_path = data_dir / "labels.npy"
    if not images_path.exists() or not labels_path.exists():
        return []
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    records: List[Dict] = []
    cache_dir = Path(__file__).resolve().parents[3] / "outputs" / "conic_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    class_map = {
        1: "Neutrophil",
        2: "Epithelial",
        3: "Lymphocyte",
        4: "Plasma",
        5: "Eosinophil",
        6: "Connective tissue",
    }
    for idx in range(images.shape[0]):
        img_path = cache_dir / f"conic_{idx:05d}.png"
        if not img_path.exists():
            Image.fromarray(images[idx]).save(img_path)
        inst_map = labels[idx, :, :, 0]
        cls_map = labels[idx, :, :, 1]
        instances: List[Dict] = []
        for inst_id in _iter_unique_values(inst_map):
            mask = (inst_map == inst_id).astype(np.uint8)
            cls_vals = cls_map[mask.astype(bool)]
            if cls_vals.size == 0:
                continue
            cls_id = int(np.bincount(cls_vals).argmax())
            label = class_map.get(cls_id, "")
            if not label:
                continue
            boxes = _mask_to_bboxes(mask, min_size=4)
            for bbox in boxes:
                instances.append({"bbox": bbox, "label": label})
        if instances:
            records.append(
                {
                    "image_path": str(img_path),
                    "width": 256,
                    "height": 256,
                    "instances": instances,
                }
            )
    return records


def read_puma(root_dir: str) -> List[Dict]:
    roi_dir = Path(root_dir) / "01_training_dataset_geojson_nuclei"
    img_dir = Path(root_dir) / "01_training_dataset_tif_ROIs"
    if not roi_dir.exists() or not img_dir.exists():
        return []
    records: List[Dict] = []
    for geojson_path in roi_dir.glob("*_nuclei.geojson"):
        stem = geojson_path.name.replace("_nuclei.geojson", "")
        img_path = img_dir / f"{stem}.tif"
        if not img_path.exists():
            continue
        try:
            img, width, height = _safe_open_image(str(img_path))
        except Exception:
            continue
        with open(geojson_path, "r") as f:
            data = json.load(f)
        instances: List[Dict] = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            cls = props.get("classification", {}).get("name")
            geom = feat.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates", [])
            if not coords:
                continue
            xs = [pt[0] for pt in coords[0]]
            ys = [pt[1] for pt in coords[0]]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            bbox = _clamp_bbox(bbox, width, height)
            if not bbox:
                continue
            instances.append({"bbox": bbox, "label": cls})
        if instances:
            records.append(
                {
                    "image_path": str(img_path),
                    "width": width,
                    "height": height,
                    "instances": instances,
                }
            )
    return records


__all__ = [
    "read_ihc_tlymphoctype",
    "read_midog_json",
    "read_midog21",
    "read_breast_nucls",
    "read_monusac_processed",
    "read_segpc",
    "read_nuclick",
    "read_panuke",
    "read_lizard",
    "read_conic",
    "read_puma",
]
