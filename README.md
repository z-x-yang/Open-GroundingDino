# Open-GroundingDino: Cell & Nucleus Grounding

This fork repurposes GroundingDINO for pathology: given a natural-language description (attribute-driven) plus an object target (`cell` or `nucleus`), the model detects matching instances and outputs bounding boxes. We curate cell/nucleus datasets, generate discriminative prompts from a structured attribute table, tile large slides into 256×256 patches, and fine-tune GroundingDINO-T to ground complex biological concepts (e.g., “object=nucleus; Epigenetic State=chromatin remodeling; Surface Markers=CD3”).

## Quick Start

```bash
conda activate openground
pip install -r requirements.txt
cd models/GroundingDINO/ops && python setup.py build install && python test.py
cd ../../../
python tools/build_cell_grounding.py --datasets breast_nucls breast_midog21 ihc_tlymphoctype mix_midog22_b
torchrun --nproc_per_node=1 main.py --config-file config/cfg_odvg.py \
  --datasets config/datasets_cell_debug.json --output-dir outputs/debug_cell --epochs 1 --batch_size 2 --amp
```

## Repository Map

- `groundingdino/`, `models/GroundingDINO/`: upstream training/inference stack.
- `datasets/path-sam/`: medical datasets, preprocessors, attribute tables (`CellAttributesV2.csv`), mappings, and build scripts.
- `outputs/`: generated ODVG jsonl, COCO eval splits, and patch tiles.
- `tests/`: pytest guards for prompt/data sanity.
- `tools/build_cell_grounding.py`: entry point that reads raw datasets, tiles slides (256 px, 10% overlap), filters to valid cell/nucleus labels, generates attribute-grounded prompts, and exports ODVG/COCO files.

## Data Preparation

1. **Raw data**: place cell/nucleus detection datasets under the locations specified in `datasets/path-sam/utils/cfg.py`. The initial pilot uses:
   - `breast_nucls` (nucleus bboxes),
   - `breast_midog21` (cell mitosis bboxes),
   - `ihc_tlymphoctype` (cell bboxes with lymphocyte/tumor classes),
   - `mix_midog22_b` (cell mitosis bboxes).
2. **Mappings**: edit `datasets/path-sam/mappings.csv` to keep only cell/nucleus categories and map them to canonical names in the attribute sheet. Non-cell annotations must be flagged `Keep=False`.
3. **Attribute prompts**: `datasets/path-sam/build/prompt_generator.py` reads `CellAttributesV2.csv` via `xlsx_utils.load_cell_attributes_csv`, then generates compact, unique descriptions that stay below the BERT 256-token limit.
4. **Tiling**: `patcher.project_and_filter_bboxes` intersects annotations with 256×256 tiles. Tiles use ≥25% visibility and ≥4 px per dimension; overlapping stride defaults to 230 (~10% overlap).

## Training & Evaluation

### Configs
- `config/cfg_odvg.py`: baseline GroundingDINO-T hyperparameters (Swin-T backbone, 900 queries, text length 256).
- `config/datasets_cell_debug.json`: mini config referencing the four pilot datasets’ ODVG/COCO exports under `outputs/`.

### Commands
- **Single GPU debug**:
  ```bash
  torchrun --nproc_per_node=1 main.py \
    --config-file config/cfg_odvg.py \
    --datasets config/datasets_cell_debug.json \
    --output-dir outputs/debug_cell \
    --epochs 2 --batch_size 2 --amp
  ```
- **SLURM multi-GPU**:
  ```bash
  sbatch scripts/train_cell_debug.slurm
  ```
  `train_slurm.sh` / `test_slurm.sh` are available for full-scale runs. Update `config/datasets_cell_full.json` once all datasets are exported.

- **Evaluation** (`test_coco_eval_bbox` in logs reports COCO-style AP for the validation COCO splits—this is the grounding AP we track):
  ```bash
  bash test_dist.sh $GPU_NUM config/cfg_odvg.py config/datasets_cell_debug.json outputs/debug_cell
  ```

## Testing & Debugging

- `pytest tests` validates ODVG/CAPTION structure, ensures attribute-driven prompts reference `datasets/path-sam/mappings.csv`, and checks generated COCO files.
- Use `python tools/build_cell_grounding.py --dryrun` to inspect mappings and prompt lengths before exporting.
- Inspect `outputs/*/train.log` for loss curves; BERT text lengths >256 indicate prompt builder issues.

## Roadmap

1. Finalize the four pilot datasets and ensure balanced cell vs nucleus prompts.
2. Run short SLURM jobs (2–3 epochs) to verify convergence and evaluate on validation tiles.
3. Scale to full datasets, log metrics (mAP, recall per cell type), and maintain experiment notes under `docs/experiments/`.
4. Extend to segmentation-to-box conversions and additional datasets after the pilot stabilizes.

For questions or contributions, update `AGENTS.md` and the experiment log when you touch the data pipeline or training scripts.
