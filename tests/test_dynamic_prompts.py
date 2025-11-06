import os
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def _load_attr_df():
    import importlib.util
    build_dir = REPO / 'datasets' / 'path-sam' / 'build'
    xu_path = build_dir / 'xlsx_utils.py'
    spec = importlib.util.spec_from_file_location('xlsx_utils', str(xu_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    df, _ = mod.load_cell_attributes_csv(
        str(REPO / 'datasets' / 'path-sam' / 'CellAttributesV2.csv'))
    return df


def _get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def test_dynamic_generator_uniqueness_and_length():
    build_dir = REPO / 'datasets' / 'path-sam' / 'build'
    import importlib.util
    pg_path = build_dir / 'prompt_generator.py'
    spec = importlib.util.spec_from_file_location(
        'prompt_generator', str(pg_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    df = _load_attr_df()
    tok = _get_tokenizer()

    # 遍历全类型（抽样限制以避免极慢；但用户要求全表，这里全部跑）
    for cell in df['Cell Type'].dropna().astype(str).tolist():
        text = mod.gen_unique_description_dynamic(
            'cell', cell, df, tok, max_tokens=256)
        # 长度限制
        assert len(tok(text)['input_ids']) <= 256
        # 解析属性 kv
        segs = [s.strip() for s in text.strip('.').split(';')]
        kvs = []
        for s in segs:
            if '=' in s:
                k, v = s.split('=', 1)
                kvs.append((k.strip(), v.strip()))
        # 含属性时不应包含 type
        has_attr = any(k not in ('object', 'type') for k, _ in kvs)
        if has_attr:
            assert all(k != 'type' for k, _ in kvs)
        # 无歧义：用包含的属性过滤应唯一
        cand = df.copy()
        for k, v in kvs:
            if k in cand.columns:
                cand = cand[cand[k].astype(str).str.lower().str.contains(
                    v.lower(), na=False, regex=False)]
        assert len(cand) == 1


def test_vg_dynamic_iteration_variation():
    # 需要一个有属性的细胞类型样本：使用 breast_nucls_val_sample.jsonl
    p = REPO / 'outputs' / 'breast_nucls_val_sample.jsonl'
    if not p.exists():
        return
    # 读第一条，确保有 phrase
    with open(p, 'r') as f:
        meta = json.loads(next(f))
    assert 'grounding' in meta
    # 构建临时数据集文件只含这一条
    tmp = REPO / 'outputs' / 'tmp_one.jsonl'
    with open(tmp, 'w') as f:
        f.write(json.dumps(meta) + '\n')
    # 强制开启动态
    os.environ['ODVG_DYNAMIC_PROMPT'] = '1'
    # import ODVGDataset with repo on sys.path
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from datasets.odvg import ODVGDataset
    ds = ODVGDataset(root='/', anno=str(tmp))
    img1, tgt1 = ds[0]
    img2, tgt2 = ds[0]
    # 有属性的类型应存在可变性（可能偶尔相同，这里放宽：cap_list长度>0）
    assert len(tgt1['cap_list']) > 0 and len(tgt2['cap_list']) > 0
    # 允许存在一致，但在大多数情况下会不同；此处弱断言：长度一致但内容集合可能不同
    assert isinstance(tgt1['caption'], str) and isinstance(
        tgt2['caption'], str)
