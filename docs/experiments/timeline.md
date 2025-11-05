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
