# Experiment Timeline

## 2025-11-03

- **Repository onboarding**
  - Replaced upstream README with project-specific goals (cell/nucleus grounding).
  - Updated `AGENTS.md` to document GPU workflows and prompt length constraints.
  - Added `config/datasets_cell_debug.json` combining breast_nucls, MIDOG21/22, mix_midog22_b.
- **Prompt/data pipeline**
  - Reworked `datasets/path-sam/build/prompt_generator.py` to cap attribute count and enforce `<256` tokens.
  - Regenerated ODVG/COCO exports; prompt stats verified (breast_nucls avg ~47 tokens).
- **Testing**
  - `pytest tests` (dynamic prompt + data pipeline) passes.
- **Initial training**
  - `srun` 1 epoch debug run → `outputs/debug_cell_srun` (AP ~0.006).
  - Documented run in `docs/experiments/debug_cell_srun.md`.

## 2025-11-03 (evening)

- **Longer training sweeps**
  - `outputs/cell_ft_smoke`: multi-epoch tuning (best AP ~0.045 @ epoch 2).
  - Metrics logged to `docs/experiments/cell_ft_smoke.md/json`.

## 2025-11-04

- **Major fine-tune (`cell_ft_tune`)**
  - Torchrun (2 GPUs) job via `jobs/cell_ft_train.slurm` (epochs=20, batch size=4, pretrained Swin-T).
  - Best validation AP ≈0.071 @ epoch 5; final AP ≈0.065.
  - Added `docs/experiments/cell_ft_tune.md` summarizing metrics.
  - Pruned redundant checkpoints; kept `checkpoint_best_regular.pth` and latest `checkpoint.pth`.
- **Documentation**
  - README note clarifying “COCO AP” is grounding AP from COCO-format eval.
  - timeline established (this file).

## 2025-11-04 (midday)

- **Extended training**
  - `jobs/cell_ft_train.slurm` now runs torchrun 2×GPU, 30 epochs, batch size 6, warmup 2, lr=4e-5 (backbone 4e-6). Job 368276 launched; once logs are available we’ll append metrics here.
- **Visualization scaffolding**
  - Added `scripts/visualize_cells.py` and `scripts/compare_predictions.py`.
  - Built CUDA ops on a GPU node (MultiScaleDeformableAttention) and generated side-by-side overlays:
    - `visuals/best_ckpt/success_case.png/json` (prompt hits multiple nuclei accurately, IoU ≈0.75–0.94).
    - `visuals/best_ckpt/failure_case.png/json` (over-specified prompt; only one nucleus matched, several FP boxes).
    - `visuals/best_ckpt/midog_success.png/json` (MIDOG cell tile, two detections covering the mitosis GT).
  - Copied对应原始 patch与大图：`visuals/best_ckpt/full_tiles/*_tile.png`（256×256 训练输入）以及 `*_source.png`（来自 `/n/lw_groups/.../nucls/eval/rgb/…` 的大图窗口）。MIDOG 原图暂未定位，仅保留 tile。
  - Threshold说明：`text_threshold` 控制短语截断，可观察哪些属性 drives detection；当前用 0.25 保持简洁。

## Outstanding tasks

1. Regain access to `tlymph` raw (IHC) to regenerate ODVG/COCO and include nucleus/cell splits.
2. After job 368276 completes, evaluate best checkpoint, update this timeline with metrics, and create qualitative visualizations.
3. Consider further tuning (longer schedule, optimizer tweaks) if AP needs to exceed current ~0.07 baseline.

## 2025-11-04 (late)

- **Dataset audit**: reviewed raw Path-SAM sources (nucls, MIDOG21/22, Lizard, CoNIC, panuke, Hemato_Data, SEGPC, PUMA, SegPath) to capture image/instance counts and label vocab for bbox/instance tasks.
- **Mapping prep**: aligned observed labels with `datasets/path-sam/CellAttributesV2.csv`, logging gaps (e.g., fibroblast ↔ CAF, stromal classes, combined lymph/plasma) for prompt design follow-up.

- **Path-SAM ODVG v2**: regenerated breast_nucls/midog21/midog22 ODVG exports using `CellAttributesV2.csv`; mapped tumor/fibroblast nuclei → CAF/invasive rows; produced COCO val files for regression tests.

- **Ops / GPU workflow doc**: documented conda env (`openground`), SLURM srun/sbatch usage, and interactive debugging flow in AGENTS.md for consistent GPU usage.

- **Panuke ODVG export**: converted panuke instance masks to bbox ODVG (`panuke_train/val*.jsonl` + COCO val), images staged under `outputs/panuke_images/`, attributes sourced from `CellAttributesV2.csv`.

- **ODVG exports**: automated Lizard, CoNIC, MoNuSAC, PUMA, NuClick (Hemato) conversions to ODVG/COCO via `tools/build_cell_grounding.py`; segpc pending (bbox unpack error under investigation).

- **SEGPC ODVG**: fixed mask decoding (strip color channels) and exported segpc_train/val jsonl + COCO val; updated datasets_path_sam_full.json to include segpc splits.

- **Train job 368418**: paused after epoch 0 (~18 min) due to tokenizer download throttling (HF 429). Need to preload bert-base-uncased tokenizer locally or cache before requeue.

## 2025-11-05

- **SEGPC verification**: Re-audited `segpc_train/val.jsonl` (9,630 train, 1,028 val records) to confirm bbox integrity before unified training.
- **HF cache & SLURM script**: cached `bert-base-uncased` snapshot locally and updated `jobs/path_sam_ft.slurm` to point at the cache while avoiding online lookups; job 368418 progressed without 429 stalls (epoch-1 loss ≈253 vs 6.6k previously).
- **Per-dataset evaluator**: added `tools/eval_path_sam_all.py` plus `jobs/path_sam_eval.slurm` to sweep every ODVG val split with the new checkpoint; job 368423 produced `outputs/path_sam_ft_full/eval_runs/*/log.txt` and an aggregated `summary.json`.
- **Dataset stats snapshot**: wrote `docs/datasets/path_sam_dataset_stats.json` capturing image/box counts for each ODVG split (helps cross-check future re-exports).
- **Unified training (job 368418)**: 5-epoch torchrun finished (≈3 h); mAP peaked at epoch 1 (0.0494) before declining after LR drop (epoch 4 mAP ≈2.7e-4). Latest loss steady around 0.19 bbox / 24 CE with LR=1e-5.
- **Post-train eval (job 368427)**: re-ran evaluator on `checkpoint_best_regular.pth`; top APs—nuclick 0.126, breast_nucls 0.051, puma 0.045, panuke 0.021, segpc 0.0078; CoNIC/Lizard remain <0.005, MIDOG21/22 ≈0.
