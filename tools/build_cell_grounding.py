import os
import json
import random
import importlib.util
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image


def _load_module(name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
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

    xlsx_path = repo_root / "datasets" / "path-sam" / "CellType2Attributes.xlsx"
    df, cols = xlsx_utils.load_cell_attributes(str(xlsx_path))
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
        # write CSV (overwrite)
        full.to_csv(mappings_csv, index=False)
        print(
            f"[build] wrote initial mappings to {mappings_csv} (rows={len(full)})")
    else:
        print("[build] no mappings generated")
    # Minimal IHC sample export smoke (no I/O yet, just count)
    try:
        ihc_dir = str((repo_root / "datasets" / "path-sam"))
        recs = readers.read_ihc_tlymphoctype(ihc_dir)[:3]
        print(f"[build] ihc sample records: {len(recs)}")
    except Exception as e:
        print("[build] ihc reader failed:", e)
    print("[build] pipeline scaffolding ready")


if __name__ == "__main__":
    main()
