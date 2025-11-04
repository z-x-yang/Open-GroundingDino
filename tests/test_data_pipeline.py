import json
import os
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / 'outputs'


def _read_jsonl(path: Path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line)


def test_odvg_structure_and_prompts_exist():
    # pick any existing ODVG files if present
    candidates = [
        OUTPUTS / 'midog22_train.jsonl',
        OUTPUTS / 'midog22_val.jsonl',
        OUTPUTS / 'midog21_train_sample.jsonl',
        OUTPUTS / 'midog21_val_sample.jsonl',
        OUTPUTS / 'breast_nucls_train_sample.jsonl',
        OUTPUTS / 'breast_nucls_val_sample.jsonl',
    ]
    files = [p for p in candidates if p.exists()]
    assert files, 'No ODVG jsonl found in outputs/'

    # sample up to 50 records for speed
    checked = 0
    for p in files:
        for rec in _read_jsonl(p):
            assert 'filename' in rec and 'grounding' in rec
            g = rec['grounding']
            assert 'caption' in g and isinstance(g['caption'], str)
            assert 'regions' in g and isinstance(g['regions'], list)
            # every region has bbox and phrase
            for r in g['regions']:
                assert 'bbox' in r and len(r['bbox']) == 4
                assert 'phrase' in r and isinstance(r['phrase'], str)
                # phrase should follow compact template prefix
                assert 'object=' in r['phrase'] and 'type=' in r['phrase']
            # caption is concatenation; length guard
            assert len(g['caption']) <= 2000
            checked += 1
            if checked >= 50:
                return


def test_coco_structure_if_present():
    candidates = [
        OUTPUTS / 'midog22_val_coco.json',
        OUTPUTS / 'midog21_val_coco.json',
        OUTPUTS / 'breast_nucls_val_coco.json',
    ]
    files = [p for p in candidates if p.exists()]
    if not files:
        return
    for p in files:
        with open(p, 'r') as f:
            data = json.load(f)
        assert 'images' in data and 'annotations' in data and 'categories' in data
        assert isinstance(data['images'], list)
        assert isinstance(data['annotations'], list)
        assert isinstance(data['categories'], list)
        # minimal sanity
        if data['images']:
            img = data['images'][0]
            for k in ['id', 'file_name', 'width', 'height']:
                assert k in img
        if data['annotations']:
            anno = data['annotations'][0]
            for k in ['id', 'image_id', 'category_id', 'bbox']:
                assert k in anno


def test_mappings_applied_in_prompts():
    # ensure mappings.csv driven phrases present when files exist
    map_csv = REPO / 'datasets' / 'path-sam' / 'mappings.csv'
    assert map_csv.exists(), 'mappings.csv missing'
    df = pd.read_csv(map_csv)
    known_std = set(str(s).strip()
                    for s in df['StandardCellName'] if isinstance(s, str) and s.strip())
    if not known_std:
        return
    # scan one odvg file
    p = OUTPUTS / 'breast_nucls_val_sample.jsonl'
    if not p.exists():
        return
    for rec in _read_jsonl(p):
        for r in rec['grounding']['regions']:
            phrase = r['phrase']
            # type should appear in phrase (case-insensitive)
            pl = phrase.lower()
            assert any(std.lower() in pl for std in known_std)
        break
