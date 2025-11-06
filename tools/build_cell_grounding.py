import os
import json
import random
import importlib.util
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image


def _load_module(name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # progress bar util (no hard dependency)
    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        def _tqdm(x, **kwargs):
            return x
    # optional threading utils for I/O-bound parallelism
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
    except Exception:
        ThreadPoolExecutor = None
        as_completed = None
        threading = None

    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "datasets" / "path-sam" / "build"

    xlsx_utils = _load_module("xlsx_utils", str(base_dir / "xlsx_utils.py"))
    mapping_builder = _load_module(
        "mapping_builder", str(base_dir / "mapping_builder.py"))
    patcher = _load_module("patcher", str(base_dir / "patcher.py"))
    odvg_exporter = _load_module(
        "odvg_exporter", str(base_dir / "odvg_exporter.py"))
    coco_exporter = _load_module(
        "coco_exporter", str(base_dir / "coco_exporter.py"))
    prompt_gen = _load_module("prompt_generator", str(
        base_dir / "prompt_generator.py"))
    readers = _load_module("readers", str(base_dir / "readers.py"))
    cfg_mod = _load_module("cfg_mod", str(
        repo_root / "datasets" / "path-sam" / "utils" / "cfg.py"))

    xlsx_path = repo_root / "datasets" / "path-sam" / "CellType2Attributes.xlsx"
    df, cols = xlsx_utils.load_cell_attributes(str(xlsx_path))
    print(f"[build] start: load attributes from {xlsx_path}")
    print(
        f"[build] loaded attributes: rows={len(df)}, columns={len(cols)}; has Cell Type={ 'Cell Type' in cols }")
    # quick smoke generate description
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cell_name = str(df.iloc[0]["Cell Type"]).strip()
        desc = prompt_gen.gen_unique_description(
            "nucleus", cell_name, df, tokenizer, max_tokens=256)
        print("[build] sample description:", desc)
        print("[build] token count:", len(tokenizer(desc)["input_ids"]))
    except Exception as e:
        print("[build] prompt generation failed:", e)

    # Build initial label mappings for 4 datasets
    import ast

    def load_breast_nucls_labels(py_path: Path) -> List[str]:
        text = py_path.read_text(encoding='utf-8')
        # crude extraction of _LABEL_DEF dict
        start = text.find("_LABEL_DEF")
        if start == -1:
            return []
        brace = text.find("{", start)
        end = text.find("}\n", brace)
        if brace == -1 or end == -1:
            return []
        dict_text = text[brace:end+1]
        try:
            # replace comments
            cleaned = "\n".join([l.split('#')[0]
                                for l in dict_text.splitlines()])
            d = ast.literal_eval(cleaned)
            vals = [str(v).strip() for k, v in d.items() if isinstance(v, str)]
            # filter known non-cell markers
            ignore = {"fov", "unlabled", "unknown"}
            return [v for v in vals if v not in ignore]
        except Exception:
            return []

    mappings_csv = repo_root / "datasets" / "path-sam" / "mappings.csv"
    print(f"[build] start: build initial mappings → {mappings_csv}")
    rows = []
    # 1) breast_nucls (nucleus)
    nucls_py = repo_root / "datasets" / "path-sam" / \
        "preprocess_seg" / "breast_nucls_b.py"
    nucls_labels = load_breast_nucls_labels(nucls_py)
    if nucls_labels:
        df_map = mapping_builder.build_initial_mappings(
            df, "breast_nucls", nucls_labels, "nucleus")
        rows.append(df_map)
    else:
        print("[build] WARN: failed to parse breast_nucls labels")
    # 2) breast_midog21 (cell)
    df_map = mapping_builder.build_initial_mappings(
        df, "breast_midog21", ["mitoses"], "cell")
    rows.append(df_map)
    # 3) ihc_tlymphoctype (cell)
    ihc_labels = ["CD3+ immune cell", "Tumor cell", "Diverse", "Other cell"]
    df_map = mapping_builder.build_initial_mappings(
        df, "ihc_tlymphoctype", ihc_labels, "cell")
    rows.append(df_map)
    # 4) mix_midog22_b (cell)
    df_map = mapping_builder.build_initial_mappings(
        df, "mix_midog22_b", ["mitoses"], "cell")
    rows.append(df_map)

    if rows:
        full = rows[0]
        for r in rows[1:]:
            full = pd.concat([full, r], ignore_index=True)
        # Heuristic mapping overrides for non-exact names
        # - CD3+ immune cell -> Lymphocyte (cell)
        # - Tumor cell -> Epithelial cell (cell) [neoplastic]
        # - mitoses -> mitoses (keep original as canonical if not in xlsx)
        # - Diverse/Other cell -> drop

        def _override(row):
            lab = str(row["OriginalLabel"]).strip()
            if lab.lower() == "cd3+ immune cell":
                row["StandardCellName"] = "Lymphocyte"
                row["Keep"] = True
                row["Note"] = "heuristic map"
            elif lab.lower() == "tumor cell":
                # map to Epithelial cell as surrogate of neoplastic epithelium
                row["StandardCellName"] = "Epithelial cell"
                row["Keep"] = True
                row["Note"] = "heuristic map (neoplastic)"
            elif lab.lower() in ("diverse", "other cell"):
                row["StandardCellName"] = ""
                row["Keep"] = False
                row["Note"] = "drop ambiguous"
            elif lab.lower() == "mitoses":
                # map到标准细胞名
                row["StandardCellName"] = "Mitosis (mitotic cell)"
                row["Keep"] = True
                row["Note"] = "map mitoses to Mitosis (mitotic cell)"
            # nucleus labels from breast_nucls
            else:
                manual = {
                    'tumor': 'Epithelial cell',
                    'fibroblast': 'Fibroblast',
                    'mitotic figure': 'mitoses',
                    'vascular endothelium': 'Endothelial cell',
                    'myoepithelium': 'Myoepithelial cell',
                    'apoptotic body': 'Apoptotic cell',
                    'ductal epithelium': 'Epithelial cell',
                }
                k = lab.lower()
                if k in manual:
                    row["StandardCellName"] = manual[k]
                    row["Keep"] = True
                    row["Note"] = "manual map"
            return row

        full = full.apply(_override, axis=1)
        # write CSV (overwrite)
        full.to_csv(mappings_csv, index=False)
        print(
            f"[build] wrote initial mappings to {mappings_csv} (rows={len(full)})")
    else:
        print("[build] no mappings generated")
    # Load mappings for runtime usage
    runtime_map = {}
    try:
        df_map = pd.read_csv(mappings_csv)
        for _, row in df_map.iterrows():
            key = (str(row.get('Dataset', '')).strip().lower(),
                   str(row.get('OriginalLabel', '')).strip().lower())
            runtime_map[key] = {
                'keep': bool(row.get('Keep', True)),
                'std': str(row.get('StandardCellName', '')).strip()
            }
        print(f"[build] loaded mappings runtime entries: {len(runtime_map)}")
    except Exception as e:
        print('[build] WARN: failed to load mappings.csv at runtime:', e)
    # Minimal sample export for two datasets if raw exists
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        VLDATA_RAW = getattr(cfg_mod, 'VLDATA_RAW')
        VLDATA_PROCESS = getattr(cfg_mod, 'VLDATA_PROCESS')
        os.makedirs(VLDATA_PROCESS, exist_ok=True)

        target_mpp = float(getattr(cfg_mod, 'TARGET_MPP', os.environ.get('PATH_SAM_TARGET_MPP', 0.5)))
        default_dataset_mpp = {
            'breast_midog21': 0.25,
            'mix_midog22_b': 0.25,
            'mix_monusac20': 0.5,
            'bone_segpc': 0.3,
            'blood_nuclick': 0.25,
            'mix_panuke': 0.5,
            'skin_puma': 0.5,
            'colon_lizard': 0.5,
            'colon_conic': 0.5,
            'breast_nucls': 0.5,
        }
        dataset_mpp_overrides: Dict[str, float] = {}
        mpp_cfg = getattr(cfg_mod, 'PATH_SAM_MPP_JSON', None)
        if not mpp_cfg:
            mpp_cfg = repo_root / 'config' / 'path_sam_mpp.json'
        mpp_cfg_path = Path(mpp_cfg)
        if mpp_cfg_path.exists():
            try:
                with open(mpp_cfg_path, 'r') as f:
                    raw = json.load(f)
                for key, val in raw.items():
                    try:
                        dataset_mpp_overrides[str(key).strip().lower()] = float(val)
                    except Exception:
                        continue
                print(f"[build] loaded path_sam_mpp entries: {len(dataset_mpp_overrides)}")
            except Exception as e:
                print(f"[build] WARN: failed to load path_sam_mpp.json: {e}")

        def lookup_dataset_mpp(dataset_tag: str) -> float:
            tag = (dataset_tag or '').strip().lower()
            if not tag:
                return 0.0
            if tag in dataset_mpp_overrides:
                return dataset_mpp_overrides[tag]
            return default_dataset_mpp.get(tag, 0.0)

        def map_label_to_standard(label: str, object_type: str, dataset: str = '') -> str:
            lab = str(label).strip()
            dskey = (str(dataset).strip().lower(), lab.lower())
            if dskey in runtime_map:
                if not runtime_map[dskey]['keep']:
                    return ''
                std = runtime_map[dskey]['std']
                if std:
                    return std
            if lab.lower() == 'cd3+ immune cell':
                return 'Lymphocyte'
            if lab.lower() == 'tumor cell':
                return 'Epithelial cell'
            if lab.lower() == 'mitoses':
                return 'Mitosis (mitotic cell)'
            # manual mapping for nucleus categories from breast_nucls
            manual = {
                'tumor': 'Epithelial cell',
                'fibroblast': 'Fibroblast',
                'mitotic figure': 'Mitosis (mitotic cell)',
                'vascular endothelium': 'Endothelial cell',
                'myoepithelium': 'Myoepithelial cell',
                'apoptotic body': 'Apoptotic cell',
                'ductal epithelium': 'Epithelial cell',
            }
            key = lab.lower()
            if key in manual:
                return manual[key]
            # default: discard unknown (return empty)
            return ''

        def export_odvg_sample(recs: List[Dict], object_type: str, out_name: str, dataset_tag: str = ''):
            # choose writable root for outputs: prefer repo_root/outputs
            out_root = str(repo_root / 'outputs')
            os.makedirs(out_root, exist_ok=True)
            # if VLDATA_PROCESS is writable, use it instead
            try:
                if os.path.isdir(VLDATA_PROCESS) and os.access(VLDATA_PROCESS, os.W_OK):
                    out_root = VLDATA_PROCESS
            except Exception:
                pass
            out_jsonl = os.path.join(out_root, f"{out_name}.jsonl")
            # cache for descriptions to reduce tokenizer calls
            desc_cache: Dict[str, str] = {}

            def get_desc_cached(std: str) -> str:
                key = f"{object_type}::{std}"
                if key in desc_cache:
                    return desc_cache[key]
                try:
                    text = prompt_gen.gen_unique_description(
                        object_type, std, df, tokenizer, max_tokens=256)
                except Exception:
                    text = f"object={object_type}; type={std}."
                desc_cache[key] = text
                return text

            # generate tile boxes using image size only (avoid numpy conversion)
            def tiles_by_size(width: int, height: int, tile_size: int = 256, stride: int = 230):
                boxes = []
                y = 0
                while y <= max(0, height - tile_size):
                    x = 0
                    while x <= max(0, width - tile_size):
                        x1 = min(x + tile_size, width)
                        y1 = min(y + tile_size, height)
                        xx = x
                        yy = y
                        if (x1 - x) < tile_size and width >= tile_size:
                            xx = width - tile_size
                            x1 = width
                        if (y1 - y) < tile_size and height >= tile_size:
                            yy = height - tile_size
                            y1 = height
                        boxes.append((xx, yy, x1, y1))
                        x += stride
                    y += stride
                br = (max(0, width - tile_size),
                      max(0, height - tile_size), width, height)
                if boxes and boxes[-1] != br:
                    boxes.append(br)
                seen = set()
                uniq = []
                for b in boxes:
                    if b not in seen:
                        seen.add(b)
                        uniq.append(b)
                return uniq

            writer_lock = threading.Lock() if threading else None
            f_jsonl = open(out_jsonl, 'w')

            def write_record(rec: Dict):
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                if writer_lock:
                    with writer_lock:
                        f_jsonl.write(line)
                else:
                    f_jsonl.write(line)

            def process_one_image(rec: Dict) -> int:
                img_path = rec['image_path']
                if not os.path.exists(img_path):
                    return 0
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception:
                    return 0
                width, height = img.size
                instances = list(rec.get('instances', []))

                # harmonise magnification if metadata is available
                src_mpp = lookup_dataset_mpp(dataset_tag)
                scale_factor = 1.0
                if src_mpp and target_mpp and src_mpp > 0 and target_mpp > 0:
                    scale_factor = src_mpp / target_mpp
                if abs(scale_factor - 1.0) > 1e-3:
                    new_w = int(round(width * scale_factor))
                    new_h = int(round(height * scale_factor))
                    if new_w >= 256 and new_h >= 256:
                        resample = Image.BICUBIC if scale_factor < 1.0 else Image.BILINEAR
                        img = img.resize((new_w, new_h), resample=resample)
                        width, height = img.size
                        scaled_instances = []
                        for ins in instances:
                            x0, y0, x1, y1 = ins['bbox']
                            scaled_instances.append({
                                **ins,
                                'bbox': (
                                    int(round(x0 * scale_factor)),
                                    int(round(y0 * scale_factor)),
                                    int(round(x1 * scale_factor)),
                                    int(round(y1 * scale_factor)),
                                )
                            })
                        instances = scaled_instances
                    else:
                        scale_factor = 1.0
                        print(f"[build] skip scaling for {dataset_tag}: scaled size {new_w}x{new_h} < 256")

                tiles = tiles_by_size(width, height, 256, 230)[:50]
                labels = list({ins['label'] for ins in instances})
                # apply mappings; drop instances with keep=False (std=='')
                mapped = {}
                for lab in labels:
                    std = map_label_to_standard(lab, object_type, dataset_tag)
                    if not std:
                        continue
                    mapped[lab] = get_desc_cached(std)
                if not mapped:
                    return 0
                out_img_dir = os.path.join(out_root, 'patches', out_name)
                os.makedirs(out_img_dir, exist_ok=True)
                produced = 0
                bbs = [ins['bbox'] for ins in instances]
                for (x0, y0, x1, y1) in tiles:
                    proj = patcher.project_and_filter_bboxes(
                        bbs, (x0, y0, x1, y1), 0.25, 4)
                    if not proj and bbs:
                        inters = []
                        for bb in bbs:
                            ix0 = max(bb[0], x0)
                            iy0 = max(bb[1], y0)
                            ix1 = min(bb[2], x1)
                            iy1 = min(bb[3], y1)
                            if ix1 > ix0 and iy1 > iy0:
                                inters.append(
                                    (ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
                        if inters:
                            proj = [inters[0]]
                    if not proj:
                        continue
                    present_labels = []
                    regions = []
                    for ins in instances:
                        inter = (
                            max(ins['bbox'][0], x0),
                            max(ins['bbox'][1], y0),
                            min(ins['bbox'][2], x1),
                            min(ins['bbox'][3], y1)
                        )
                        if inter[2] > inter[0] and inter[3] > inter[1]:
                            local = (inter[0]-x0, inter[1]-y0,
                                     inter[2]-x0, inter[3]-y0)
                            if (local[2]-local[0]) >= 4 and (local[3]-local[1]) >= 4:
                                if ins['label'] in mapped:
                                    present_labels.append(ins['label'])
                                    regions.append(
                                        {'bbox': [local[0], local[1], local[2], local[3]], 'phrase': mapped[ins['label']]})
                    if not regions:
                        continue
                    tile_img_path = os.path.join(
                        out_img_dir, f"{Path(img_path).stem}_{x0}_{y0}.png")
                    # fast PNG save
                    img.crop((x0, y0, x1, y1)).save(
                        tile_img_path, compress_level=1, optimize=False)
                    caption = " ".join([mapped[l]
                                       for l in sorted(set(present_labels))])
                    write_record({
                        'filename': tile_img_path,
                        'height': (y1-y0),
                        'width': (x1-x0),
                        'grounding': {
                            'caption': caption if caption.endswith('.') else caption + '.',
                            'regions': regions
                        }
                    })
                    produced += 1
                return produced

            print(
                f"[build][{out_name}] begin: {len(recs)} images → tiles + jsonl")
            produced_total = 0
            if ThreadPoolExecutor is not None:
                max_workers = min(8, os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(process_one_image, rec)
                               for rec in recs]
                    for fut in _tqdm(as_completed(futures), total=len(futures), desc=f"{out_name} images"):
                        try:
                            produced_total += fut.result()
                        except Exception:
                            pass
            else:
                for rec in _tqdm(recs, desc=f"{out_name} images"):
                    produced_total += process_one_image(rec)
            f_jsonl.close()
            print(
                f"[build] {out_name} odvg_records={produced_total} (images={len(recs)}) out_root={out_root}")
            if produced_total > 0:
                print(
                    f"[build] wrote ODVG sample: {out_jsonl} count={produced_total}")

        def export_midog22_train_val():
            # read bigger pool and split 8:2
            midog_json = os.path.join(
                VLDATA_RAW, 'midog2', 'MIDOG2022_training.json')
            midog_imgd = os.path.join(VLDATA_RAW, 'midog2', 'images')
            if not (os.path.isfile(midog_json) and os.path.isdir(midog_imgd)):
                print('[build] skip full MIDOG22 train/val export (raw missing)')
                return
            pool = readers.read_midog_json(midog_json, midog_imgd)
            if not pool:
                print('[build] skip full MIDOG22 export (empty pool)')
                return
            n_train = int(len(pool) * 0.8)
            train_recs = pool[:n_train]
            val_recs = pool[n_train:]

            # helper to export to ODVG
            def export_split(recs, split_name):
                out_root = str(repo_root / 'outputs')
                os.makedirs(out_root, exist_ok=True)
                out_jsonl = os.path.join(
                    out_root, f'midog22_{split_name}.jsonl')
                # streaming writer
                writer_lock = threading.Lock() if threading else None
                f_jsonl = open(out_jsonl, 'w')
                images_coco = []
                annos_coco = []
                img_id = 1
                anno_id = 1
                print(
                    f"[build][midog22_{split_name}] begin: {len(recs)} images")

                def write_record(rec):
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    if writer_lock:
                        with writer_lock:
                            f_jsonl.write(line)
                    else:
                        f_jsonl.write(line)

                desc_cache: Dict[str, str] = {}

                def get_desc_cached(std: str) -> str:
                    key = f"cell::{std}"
                    if key in desc_cache:
                        return desc_cache[key]
                    try:
                        text = prompt_gen.gen_unique_description(
                            'cell', std, df, tokenizer, max_tokens=256)
                    except Exception:
                        text = f"object=cell; type={std}."
                    desc_cache[key] = text
                    return text

                def tiles_by_size(width: int, height: int, tile_size: int = 256, stride: int = 230):
                    boxes = []
                    y = 0
                    while y <= max(0, height - tile_size):
                        x = 0
                        while x <= max(0, width - tile_size):
                            x1 = min(x + tile_size, width)
                            y1 = min(y + tile_size, height)
                            xx = x
                            yy = y
                            if (x1 - x) < tile_size and width >= tile_size:
                                xx = width - tile_size
                                x1 = width
                            if (y1 - y) < tile_size and height >= tile_size:
                                yy = height - tile_size
                                y1 = height
                            boxes.append((xx, yy, x1, y1))
                            x += stride
                        y += stride
                    br = (max(0, width - tile_size),
                          max(0, height - tile_size), width, height)
                    if boxes and boxes[-1] != br:
                        boxes.append(br)
                    seen = set()
                    uniq = []
                    for b in boxes:
                        if b not in seen:
                            seen.add(b)
                            uniq.append(b)
                    return uniq

                produced_total = 0

                def process_one(rec: Dict) -> int:
                    nonlocal img_id, anno_id
                    img_path = rec['image_path']
                    if not os.path.exists(img_path):
                        return 0
                    try:
                        img = Image.open(img_path).convert('RGB')
                    except Exception:
                        return 0
                    width, height = img.size
                    tiles = tiles_by_size(width, height, 256, 230)[:80]
                    labels = list({ins['label'] for ins in rec['instances']})
                    mapped = {}
                    for lab in labels:
                        std = map_label_to_standard(
                            lab, 'cell', 'mix_midog22_b')
                        if not std:
                            continue
                        mapped[lab] = get_desc_cached(std)
                    if not mapped:
                        return 0
                    out_img_dir = os.path.join(
                        out_root, 'patches', f'midog22_{split_name}')
                    os.makedirs(out_img_dir, exist_ok=True)
                    produced = 0
                    bbs = [ins['bbox'] for ins in rec['instances']]
                    for (x0, y0, x1, y1) in tiles:
                        proj = patcher.project_and_filter_bboxes(
                            bbs, (x0, y0, x1, y1), 0.25, 4)
                        if not proj:
                            continue
                        present_labels = []
                        regions = []
                        for ins in rec['instances']:
                            ix0 = max(ins['bbox'][0], x0)
                            iy0 = max(ins['bbox'][1], y0)
                            ix1 = min(ins['bbox'][2], x1)
                            iy1 = min(ins['bbox'][3], y1)
                            if ix1 > ix0 and iy1 > iy0:
                                local = (ix0 - x0, iy0 - y0,
                                         ix1 - x0, iy1 - y0)
                                if (local[2]-local[0]) >= 4 and (local[3]-local[1]) >= 4:
                                    if ins['label'] in mapped:
                                        present_labels.append(ins['label'])
                                        regions.append(
                                            {'bbox': [local[0], local[1], local[2], local[3]], 'phrase': mapped[ins['label']]})
                        if not regions:
                            continue
                        tile_img_path = os.path.join(
                            out_img_dir, f"{Path(img_path).stem}_{x0}_{y0}.png")
                        img.crop((x0, y0, x1, y1)).save(
                            tile_img_path, compress_level=1, optimize=False)
                        caption = " ".join([mapped[l]
                                           for l in sorted(set(present_labels))])
                        write_record({'filename': tile_img_path, 'height': (y1-y0), 'width': (x1-x0), 'grounding': {
                            'caption': caption if caption.endswith('.') else caption + '.', 'regions': regions}})
                        if split_name == 'val':
                            # maintain COCO lists with thread safety
                            if writer_lock:
                                with writer_lock:
                                    images_coco.append(
                                        {'id': img_id, 'file_name': tile_img_path, 'width': (x1-x0), 'height': (y1-y0)})
                                    for reg in regions:
                                        x, y, xx, yy = reg['bbox']
                                        annos_coco.append({'id': anno_id, 'image_id': img_id, 'category_id': 1, 'bbox': [
                                                          x, y, xx - x, yy - y], 'area': (xx - x) * (yy - y), 'iscrowd': 0})
                                        anno_id += 1
                                    img_id += 1
                            else:
                                images_coco.append(
                                    {'id': img_id, 'file_name': tile_img_path, 'width': (x1-x0), 'height': (y1-y0)})
                                for reg in regions:
                                    x, y, xx, yy = reg['bbox']
                                    annos_coco.append({'id': anno_id, 'image_id': img_id, 'category_id': 1, 'bbox': [
                                                      x, y, xx - x, yy - y], 'area': (xx - x) * (yy - y), 'iscrowd': 0})
                                    anno_id += 1
                                img_id += 1
                        produced += 1
                    return produced

                if ThreadPoolExecutor is not None:
                    max_workers = min(8, os.cpu_count() or 4)
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futures = [ex.submit(process_one, rec) for rec in recs]
                        for fut in _tqdm(as_completed(futures), total=len(futures), desc=f"midog22 {split_name} images"):
                            try:
                                produced_total += fut.result()
                            except Exception:
                                pass
                else:
                    for rec in _tqdm(recs, desc=f"midog22 {split_name} images"):
                        produced_total += process_one(rec)
                f_jsonl.close()
                if produced_total > 0:
                    print(
                        f"[build] wrote ODVG {split_name}: {out_jsonl} count={produced_total}")
                if split_name == 'val' and images_coco:
                    categories = [{'id': 1, 'name': 'mitoses cell'}]
                    coco_out = os.path.join(out_root, 'midog22_val_coco.json')
                    coco_exporter.export_coco(
                        images_coco, annos_coco, categories, coco_out)

            export_split(train_recs, 'train')
            export_split(val_recs, 'val')

        def export_midog21_train_val():
            jsonp = os.path.join(VLDATA_RAW, 'MIDOG21', 'MIDOG.json')
            imgd = os.path.join(VLDATA_RAW, 'MIDOG21', 'images')
            if not (os.path.isfile(jsonp) and os.path.isdir(imgd)):
                print('[build] skip MIDOG21 (raw missing)')
                return
            print("[build] start: read MIDOG21")
            pool = readers.read_midog21(jsonp, imgd)
            if not pool:
                print('[build] skip MIDOG21 (empty)')
                return
            n_train = int(len(pool) * 0.8)
            train_recs, val_recs = pool[:n_train], pool[n_train:]
            export_odvg_sample(train_recs, 'cell',
                               'midog21_train_sample', 'breast_midog21')
            export_odvg_sample(
                val_recs, 'cell', 'midog21_val_sample', 'breast_midog21')
            # build COCO for val
            out_root = str(repo_root / 'outputs')
            os.makedirs(out_root, exist_ok=True)
            images_coco, annos_coco, cats = [], [], [
                {'id': 1, 'name': 'mitoses cell'}]
            img_id = 1
            ann_id = 1
            for rec in _tqdm(val_recs, desc="midog21 val images"):
                p = rec['image_path']
                if not os.path.exists(p):
                    continue
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    continue
                w, h = img.size
                images_coco.append(
                    {'id': img_id, 'file_name': p, 'width': w, 'height': h})
                for ins in rec['instances']:
                    x0, y0, x1, y1 = ins['bbox']
                    annos_coco.append({'id': ann_id, 'image_id': img_id, 'category_id': 1, 'bbox': [
                                      x0, y0, x1-x0, y1-y0], 'area': (x1-x0)*(y1-y0), 'iscrowd': 0})
                    ann_id += 1
                img_id += 1
            coco_exporter.export_coco(images_coco, annos_coco, cats, os.path.join(
                out_root, 'midog21_val_coco.json'))

        def export_breast_nucls_train_val():
            imgd = os.path.join(VLDATA_RAW, 'nucls')
            if not os.path.isdir(imgd):
                print('[build] skip breast_nucls (raw missing)')
                return
            print("[build] start: read breast_nucls")
            recs = readers.read_breast_nucls(imgd)
            if not recs:
                print('[build] skip breast_nucls (empty)')
                return
            n_train = int(len(recs)*0.8)
            train_recs, val_recs = recs[:n_train], recs[n_train:]
            export_odvg_sample(train_recs, 'nucleus',
                               'breast_nucls_train_sample', 'breast_nucls')
            export_odvg_sample(val_recs, 'nucleus',
                               'breast_nucls_val_sample', 'breast_nucls')
            # COCO val build with multiple categories mapping-aware
            out_root = str(repo_root / 'outputs')
            os.makedirs(out_root, exist_ok=True)
            cats_map = {}
            images_coco, annos_coco = [], []
            img_id = 1
            ann_id = 1
            cat_id = 1
            for rec in _tqdm(val_recs, desc="nucls val images"):
                p = rec['image_path']
                if not os.path.exists(p):
                    continue
                try:
                    img = Image.open(p).convert('RGB')
                except Exception:
                    continue
                w, h = img.size
                images_coco.append(
                    {'id': img_id, 'file_name': p, 'width': w, 'height': h})
                for ins in rec['instances']:
                    std = map_label_to_standard(
                        ins['label'], 'nucleus', 'breast_nucls')
                    if not std:
                        continue
                    name = std + ' nucleus'
                    if name not in cats_map:
                        cats_map[name] = cat_id
                        cat_id += 1
                    cid = cats_map[name]
                    x0, y0, x1, y1 = ins['bbox']
                    annos_coco.append({'id': ann_id, 'image_id': img_id, 'category_id': cid, 'bbox': [
                                      x0, y0, x1-x0, y1-y0], 'area': (x1-x0)*(y1-y0), 'iscrowd': 0})
                    ann_id += 1
                img_id += 1
            cats = [{'id': v, 'name': k} for k, v in cats_map.items()]
            coco_exporter.export_coco(images_coco, annos_coco, cats, os.path.join(
                out_root, 'breast_nucls_val_coco.json'))

        # IHC (wrap to avoid blocking MIDOG)
        try:
            ihc_root = os.path.join(VLDATA_RAW, 'tlymph')
            if os.path.isdir(ihc_root):
                print("[build] start: read ihc_tlymphoctype")
                recs = readers.read_ihc_tlymphoctype(ihc_root)[:3]
                export_odvg_sample(recs, 'cell', 'ihc_tlymphoctype_sample')
            else:
                print(f"[build] skip IHC, not found: {ihc_root}")
        except Exception as e:
            print('[build] IHC export skipped:', e)

        # MIDOG22
        midog_json = os.path.join(
            VLDATA_RAW, 'midog2', 'MIDOG2022_training.json')
        midog_imgd = os.path.join(VLDATA_RAW, 'midog2', 'images')
        try:
            if os.path.isfile(midog_json) and os.path.isdir(midog_imgd):
                print("[build] start: read MIDOG22 small sample")
                recs = readers.read_midog_json(midog_json, midog_imgd)[:3]
                export_odvg_sample(recs, 'cell', 'mix_midog22_b_sample')
            else:
                print(f"[build] skip MIDOG22, not found: {midog_json}")
        except Exception as e:
            print('[build] MIDOG22 export skipped:', e)
        # finally attempt full train/val export from MIDOG22
        try:
            export_midog22_train_val()
        except Exception as e:
            print('[build] MIDOG22 full export skipped:', e)
        # export MIDOG21 and nucls
        try:
            export_midog21_train_val()
        except Exception as e:
            print('[build] MIDOG21 export skipped:', e)
        try:
            export_breast_nucls_train_val()
        except Exception as e:
            print('[build] nucls export skipped:', e)
    except Exception as e:
        print("[build] sample export failed:", e)
        print("[build] pipeline scaffolding ready")


if __name__ == "__main__":
    main()
