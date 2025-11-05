# Repository Guidelines

Follow these notes so changes land cleanly and models stay reproducible.

## Project Structure & Module Organization
- `groundingdino/`: core training, inference, and model heads adapted from Grounding DINO.
- `models/GroundingDINO/ops/`: CUDA/C++ extensions that must be built before training; package mirrors upstream release.
- `datasets/`: COCO/ODVG readers plus the `path-sam` build pipeline; generated JSONL/COCO artifacts live under `outputs/`.
- `tools/`: dataset conversion scripts and prompt builders; `tests/`: pytest suites covering these utilities.
- `main.py`, `engine.py`, and shell helpers (`train_dist.sh`, `train_slurm.sh`, etc.) wire everything into distributed training loops.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python deps (baseline: Python 3.7, Torch 1.11, CUDA 11.3).
- `python models/GroundingDINO/ops/setup.py build install && python models/GroundingDINO/ops/test.py`: build and validate custom ops.
- `bash train_dist.sh ${GPU_NUM} config/cfg_odvg.py config/datasets_mixed_odvg.json ./logs`: distributed train; use `*_slurm.sh` variants on clusters.
- `srun -t 0-01:00 --mem=64G -c 4 --gres=gpu:1 -p gpu_dia conda run -n openground python main.py --config_file config/cfg_odvg.py --datasets config/datasets_cell_debug.json --output_dir outputs/<run> --options epochs=1 batch_size=2`: quick GPU debug run following `job_1gpu.sh` assumptions.
- `pytest tests`: run fast regression tests for data/prompt sanity; add `-k <keyword>` for focused checks.
- Every session must append activities to `docs/experiments/timeline.md` so future runs retain chronological context; treat timeline upkeep as part of any workflow.
- When producing visual results (success/failure cases, raw tiles), stash artifacts under `visuals/` and record prompts/JSON summaries so future debugging has references.

## Coding Style & Naming Conventions
- Follow PEP 8, four-space indents, and keep lines ≤120 for long training loops.
- Favor `snake_case` for functions/vars, `CamelCase` for classes, and descriptive config filenames (`cfg_odvg.py`, `datasets_midog22_mini.json`).
- Extend existing type hints in trainers and dataset builders; include concise docstrings when adding public entry points.
- Route logging through `util.misc.MetricLogger`; gate verbose prints behind `--debug`.

## Testing Guidelines
- Tests use pytest; mirror `tests/test_data_pipeline.py` structure for dataset validations.
- Supply minimal fixtures under `outputs/` for new converters and guard tests to skip when optional artefacts are absent.
- Land PRs only after `pytest tests` passes and deterministic checks cover new prompts/exporters/evaluators; regenerate ODVG jsonl via `python tools/build_cell_grounding.py` after attr/prompt changes.

## Commit & Pull Request Guidelines
- Follow Conventional Commits as in history (`feat(path-sam):`, `chore(build):`), matching scopes to touched folders.
- Squash WIP commits before pushing and document behavior changes, affected configs/datasets, and key metrics or screenshots in the PR body.
- Link related issues or experiment logs and confirm `pytest tests` plus a representative training command succeed on at least one GPU setup.

## Configuration & Data Notes
- Update `config/datasets_mixed_odvg.json` (or dataset-specific JSON) when adding data and note ODVG/COCO paths in the PR.
- Document new environment variables, keep secrets out of git, and extend `.gitignore` for temporary assets.
- Attribute prompts must stay ≤256 BERT tokens; `datasets/path-sam/build/prompt_generator.py` enforces this—report regressions if token stats exceed 120 avg / 256 max in `outputs/*.jsonl`.
