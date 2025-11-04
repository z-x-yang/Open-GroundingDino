# Debug Run: `debug_cell_srun` (2025-11-03)

- **Command**  
  `srun -t 0-01:00 --mem=64G -c 4 --gres=gpu:1 -p gpu_dia conda run -n openground python main.py --config_file config/cfg_odvg.py --datasets config/datasets_cell_debug.json --output_dir outputs/debug_cell_srun --options epochs=1 batch_size=2 --note debug_cell_srun`

- **Dataset config**: `config/datasets_cell_debug.json` (breast_nucls + MIDOG21 + MIDOG22 + mix_midog22_b)
- **Model**: GroundingDINO Swin-T, BERT-base tokenizer, text length capped at 256 tokens.
- **Key Outputs**
  - `outputs/debug_cell_srun/log.txt`: train loss ~5.3e3 (high as expected for 1 epoch); bbox loss 1.84; giou loss 1.37.
  - `outputs/debug_cell_srun/eval/000.pth`: validation stats saved after epoch 0; COCO AP ≈ 0.006 (initial baseline).
  - Checkpoints: `checkpoint0000.pth`, `checkpoint.pth`, `checkpoint_best_regular.pth`.
  - Token stats on training JSONL: `breast_nucls max=85 avg=46.7`, `MIDOG21 max=19`, `mix_midog22_b max=19`.

- **Observations**
  - Training completes without NaNs or OOM; dynamic prompts stay within 256-token limit.
  - Validation AP still near-zero; further epochs and data balancing required.
  - `ihc_tlymphoctype` data missing due to lack of raw access; upcoming work: resolve permissions or substitute dataset.

- **Next Steps**
  1. Investigate high CE loss (~270 unscaled). Consider reducing negative labels or tuning class loss weight.
  2. Add IHC dataset once permission is resolved; regenerate ODVG exports.
  3. Run 5–10 epoch SLURM job to confirm trends; log metrics under `docs/experiments`.
  4. Produce qualitative visualizations (not yet generated).
