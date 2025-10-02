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
    prompt_gen = _load_module("prompt_generator", str(base_dir / "prompt_generator.py"))

    xlsx_path = repo_root / "datasets" / "path-sam" / "CellType2Attributes.xlsx"
    df, cols = xlsx_utils.load_cell_attributes(str(xlsx_path))
    print(
        f"[build] loaded attributes: rows={len(df)}, columns={len(cols)}; has Cell Type={ 'Cell Type' in cols }")
    # quick smoke generate description
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cell_name = str(df.iloc[0]["Cell Type"]).strip()
        desc = prompt_gen.gen_unique_description("nucleus", cell_name, df, tokenizer, max_tokens=256)
        print("[build] sample description:", desc)
        print("[build] token count:", len(tokenizer(desc)["input_ids"]))
    except Exception as e:
        print("[build] prompt generation failed:", e)
    print("[build] pipeline scaffolding ready")


if __name__ == "__main__":
    main()
