# Path-SAM → ODVG Conversion Plan (2025-11-04)

This note tracks the datasets audited on the raw storage (`/n/lw_groups/hms/dbmi/yu/lab/seg_data_raw`) and how we will map each label to the canonical attributes in `datasets/path-sam/CellAttributesV2.csv`. It supersedes any earlier plans that relied on `CellType2Attributes.xlsx`.

## Canonical Attribute Source
- **Attribute table**: `datasets/path-sam/CellAttributesV2.csv`
- **Prompt builder**: `datasets/path-sam/build/prompt_generator.py` (already handles CSV-backed lookups; ensure we only feed it `CellAttributesV2.csv` data).
- **Mapping registry**: extend `datasets/path-sam/mappings.csv` for every dataset once label alignment is final.

## Dataset Inventory

### Breast NuCLS (nucleus boxes)
- **Raw**: `/n/.../seg_data_raw/nucls` (≈2.3 GB, 3 081 images, 98 127 nuclei).
- **Labels**: tumor, fibroblast, lymphocyte, plasma cell, macrophage, mitotic figure, vascular endothelium, myoepithelium, apoptotic body, neutrophil, ductal epithelium, eosinophil.
- **Attribute mapping**:
  - direct match: Epithelial cell, Lymphocyte, Plasma cell, Macrophage, Neutrophil, Endothelial cell, Apoptosis (apoptotic cell), Mitosis (mitotic cell), Eosinophil.
  - fibroblast/myoepithelium → use “Cancer-Associated Fibroblasts” or “Smooth muscle cell / Myofibroblast”.
  - tumor/ductal epithelium → “Epithelial cell” vs “Invasive cancer cells”.
- **Notes**: existing `mappings.csv` rows need to be regenerated to reference CSV attributes only.

### MIDOG21 (bbox detection)
- **Raw**: `/n/.../seg_data_raw/MIDOG21` (≈26 GB, 200 images, 4 435 annotations).
- **Labels**: mitotic figure, not mitotic figure.
- **Mapping**: positive → “Mitosis (mitotic cell)”; negatives retained as “Non_mitosis (resting cells)” for hard negatives when generating prompts.

### MIDOG22 / mix_midog22_b (bbox detection)
- **Raw**: `/n/.../seg_data_raw/midog2` (≈110 GB, 403 images, 20 552 annotations).
- **Labels**: identical to MIDOG21.
- **Mapping**: same as above.

### TIGER ROI cells (bbox + tissue masks)
- **Raw**: `/n/.../seg_data_raw/TIGER/wsirois/roi-level-annotations/tissue-cells`.
- **Stats**: 1 879 ROI tiles, 30 524 bbox annotations in `tiger-coco.json`.
- **Labels**: single class “lymphocytes and plasma cells”.
- **Mapping**: requires disambiguation—either split via heuristics or map to a composite description (attribute text should mention both lymphocyte and plasma morphology until we can separate them).

### Panuke (instance segmentation)
- **Raw**: `/n/.../seg_data_raw/panuke/part[1-3]` (≈40 GB, 7 901 tiles of 256×256).
- **Channels → classes**: Neoplastic cells, Inflammatory, Connective/Soft tissue, Dead Cells, Epithelial, Background.
- **Mapping**:
  - Neoplastic → “Invasive cancer cells”.
  - Inflammatory → “Leukocyte (general)” (captures lymphocyte-rich infiltrates).
  - Connective → “Cancer-Associated Fibroblasts”.
  - Dead Cells → “Dead cells”.
  - Epithelial → “Epithelial cell”.
- **Action**: convert per-instance masks into bboxes, discard the background channel, split 90/10 train/val with COCO val export.

### Lizard (instance segmentation + centroid)
- **Raw**: `/n/.../seg_data_raw/Lizard` (≈2.5 GB, 238 WSIs).
- **Classes**: Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue.
- **Mapping**: connective tissue → “Cancer-Associated Fibroblasts”; others match directly.
- **Status**: ODVG (`lizard_train.jsonl`, `lizard_val.jsonl`) and COCO val (`lizard_val_coco.json`) generated via `tools/build_cell_grounding.py`.

### CoNIC (tile masks)
- **Raw**: `/n/.../seg_data_raw/CoNIC/data` (≈2.3 GB, 4 981 tiles).
- **Instance map**: 6 nucleus classes encoded per instance id.
- **Mapping**: {1: neutrophil, 2: epithelial, 3: lymphocyte, 4: plasma, 5: eosinophil, 6: connective (→ “Cancer-Associated Fibroblasts”)}.
- **Status**: ODVG (`conic_train.jsonl`, `conic_val.jsonl`) and COCO val (`conic_val_coco.json`).

### MoNuSAC (WSI segmentation → cropped)
- **Raw**: `/n/.../seg_data_raw/monusac` (WSIs) + `/n/.../monusac_processed` (cropped images and masks).
- **Classes**: Epithelial, Lymphocyte, Macrophage, Neutrophil, Ambiguous.
- **Mapping**: direct for first four; ambiguous dropped.
- **Status**: ODVG (`monusac_train.jsonl`, `monusac_val.jsonl`) and COCO val (`monusac_val_coco.json`).

### PUMA (GeoJSON nuclei)
- **Raw**: `/n/.../seg_data_raw/puma` (≈16 GB, 206 GeoJSON files).
- **Classes**: nuclei_tumor, nuclei_lymphocyte, nuclei_plasma_cell, nuclei_stroma, nuclei_endothelium, nuclei_histiocyte, nuclei_melanophage, nuclei_neutrophil, nuclei_epithelium, nuclei_apoptosis.
- **Mapping**: tumor → “Invasive cancer cells”; lymphocyte/plasma/endothelium/epithelium/neutrophil direct; stroma/histiocyte/melanophage → “Macrophage”; apoptosis → “Apoptosis (apoptotic cell)”.
- **Status**: ODVG (`puma_train.jsonl`, `puma_val.jsonl`) and COCO val (`puma_val_coco.json`).

### SEGPC (cell + nucleus masks)
- **Raw**: `/n/.../seg_data_raw/segpc` (≈21 GB, 775 images, 2 633 instances).
- **Classes**: myeloma plasma cells with cytoplasm/nucleus split.
- **Mapping**: “Plasma cell”.
- **Status**: ODVG (`segpc_train.jsonl`, `segpc_val.jsonl`) and COCO val (`segpc_val_coco.json`).

### Hemato_Data (NuClick WBC)
- **Raw**: `/n/.../seg_data_raw/Hemato_Data` (≈452 MB, 1 463 images).
- **Mapping**: all instances → “Leukocyte (general)”.
- **Status**: ODVG (`nuclick_train.jsonl`, `nuclick_val.jsonl`) and COCO val (`nuclick_val_coco.json`).

### SegPath (immunostain mask pairs)
- **Raw**: `/n/.../seg_data_raw/segpath` (≈570 GB).
- **Subdatasets**: SmoothMuscle, RBC, Lymphocyte, Leukocyte, Endothelium, PlasmaCell, MyeloidCell, Epithelium.
- **Mapping**: each subdirectory already matches attribute cell types (Smooth muscle cell / Myofibroblast, Red blood cell (RBC), Lymphocyte, Leukocyte (general), Endothelial cell, Plasma cell, Myeloid cell (general), Epithelial cell).
- **Note**: extremely large; conversions must be batched to avoid storage blow-up.

### GLySAC (gastric nuclei)
- **Raw**: `/n/.../seg_data_raw/glysac_dataset` (34 annotated images).
- **Classes**: multiple gastric cancer nucleus types (IDs 1–8). Need to map to Epithelial cell, Lymphocyte, etc. (review label dictionary once finalized).

### DigestPath19
- **Raw**: `/n/.../seg_data_raw/digestpath19` (WSI access through processed directory).
- **Status**: raw masks not immediately visible; need follow-up before conversion.

### IHC T-Lymphocyte (`tlymph`)
- **Status**: directory exists but permission denied (`/n/.../seg_data_raw/tlymph`). Blocked until access is restored.

## Conversion & Training Checklist

1. **Mapping updates**
   - Expand `datasets/path-sam/mappings.csv` for every dataset above; ensure each `StandardCellName` matches a row in `CellAttributesV2.csv`.
   - Add documentation to `datasets/path-sam/mappings.md` after validation.

2. **Conversion scripts**
   - Instance segmentation → bbox: extend/modify existing preprocess scripts to emit ODVG jsonl with bounding boxes plus prompt phrases built from `CellAttributesV2.csv`.
   - Bbox datasets: regenerate ODVG jsonl ensuring validation split (e.g., 10% hold-out per dataset).
   - Store outputs under `outputs/path-sam/odvg/<dataset>` with `train.jsonl`, `val.jsonl`, and `label_map.json`.

3. **Dataset aggregation**
   - Update `config/datasets_mixed_odvg.json` (or add new config) to point to regenerated ODVG splits.
   - Record dataset weights/sampling strategy for unified fine-tuning.

4. **Training & Evaluation**
   - Fine-tune using updated ODVG mix (document hyperparameters, seeds, checkpoints).
   - Run per-dataset evaluations (COCO metrics on val splits) and store results in `outputs/<run>/metrics_<dataset>.json`.
   - Summarize qualitative results; stash example prompts/images under `visuals/`.

5. **Timeline logging**
   - Continue appending major actions to `docs/experiments/timeline.md`.

## Open Questions
- How should we handle classes without a one-to-one attribute row (e.g., connective tissue vs. CAF vs. stromal nuclei)?
- Preferred validation split ratio per dataset (default 10% vs pre-defined splits)?
- Storage budget for large conversions (SegPath ≈570 GB raw; ODVG exports may double space).
- Confirmation once `tlymph` permissions are restored.

Let me know if any of the assumptions above need adjustment before conversion begins.


### 2025-11-05 Updates
- Evaluated checkpoint `outputs/path_sam_ft_full/checkpoint_best_regular.pth` on all current ODVG val splits using `tools/eval_path_sam_all.py`; summary saved at `outputs/path_sam_ft_full/eval_runs/summary.json`.
- Latest AP highlights: breast_nucls 0.028, panuke 0.014, puma 0.027; other datasets ~0 pending longer training.
- Dataset size snapshot recorded in `docs/datasets/path_sam_dataset_stats.json` for regeneration auditing.


### 2025-11-05 Evaluation Snapshot
- Unified 5-epoch run (job 368418) completed; best checkpoint remains epoch 1 (AP=0.049).
- Latest eval (job 368427) APs: nuclick 0.126, breast_nucls 0.051, puma 0.045, panuke 0.021, segpc 0.008; other datasets <0.005.
- Action: investigate sampling/attribute prompts for low-performing MIDOG/CoNIC/Lizard before next training cycle.

### 2025-11-05 BBox Stats Audit
- Added `tools/analyze_bbox_stats.py`; stats saved to `docs/datasets/path_sam_bbox_stats.json`.
- Nucleus datasets average ≈14×14 px boxes (relative area ≈0.7%), except MIDOG21/22, MoNuSAC, NuClick with much larger spans (relative area ≥0.1) indicating potential scale mismatches.
- Cell datasets average ≈34×34 px (relative area ≈3.5%); SEGPC very large (relative area ≈0.35) and likely requires normalization or re-sampling.
