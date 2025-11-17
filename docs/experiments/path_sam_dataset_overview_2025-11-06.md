# Path-SAM 训练数据集现状 (2025-11-06)

本文独立整理目前计划纳入 Path-SAM->ODVG 训练的细胞/细胞核数据集，重点说明数据类型、标注形式、原始数据规模、最新 MPP 归一化设置（`PATH_SAM_TARGET_MPP=0.25`），以及与 `CellAttributesV2` 的标签映射状态。训练风险与后续工作也将持续在此页汇总。（2025-11-16 更新：因 MIDOG21/22 原始标注为“大块 ROI bbox”，不适合单细胞检测，已完全移出训练/评测配置。）

## 1. 训练数据集总览

- 目前纳入统一训练的 ODVG 数据集为 **8 个**（breast_nucls、lizard、conic、monusac、panuke、puma、segpc、nuclick），并已全部在 2025-11-06 重新以 **PATH_SAM_TARGET_MPP=0.25** 归一化生成（含 COCO val）。MIDOG21/22 暂存为 raw 参考数据，不再参与训练/评测。
- **细胞级（cell）** 标注 2 个（panuke、segpc），其余 8 个为 **细胞核级（nucleus）** 标注；实例分割数据统一转成 bbox + 属性描述。
- ODVG/COCO 统计来源：`outputs/*_train.jsonl`、`outputs/*_val.jsonl`、`outputs/*_val_coco.json`（重新生成于 2025-11-06）；MPP 取自 `config/path_sam_mpp.json`。
- 当前训练作业：`sbatch 368485`（jobs/path_sam_ft.slurm，4×GPU，epochs=5）正在运行，输出路径 `outputs/path_sam_ft_full`。

### 1.1 数据规模与 MPP 概览（2025-11-06 新缓存）

| 数据集 | 对象粒度 | 标注来源 | 原始数据概况 | 典型图像尺寸 (px) | ODVG 切分 (train/val 图像, bbox) | 当前配置 MPP (um/px) | `outputs/patches` tile 数量 (train+val) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| breast_nucls | nucleus | 直接 bbox | 原始 3 081 张 ROI (~2.3 GB) | PNG ROI ~380×370 | 1 203 / 377 图像；14 301 / 3 860 bbox | 0.25 | 1 580 |
| lizard | nucleus | 实例分割→bbox | 238 张 WSI (~2.5 GB) | 500–1500² PNG | 7 584 / 910 图像；243 181 / 30 631 bbox | 0.5 | 8 494 |
| conic | nucleus | 实例分割→bbox | 4 981 张 tile (~2.3 GB) | 256² NPY | 21 249 / 2 368 图像；665 004 / 74 906 bbox | 0.5 | 23 617 |
| monusac | nucleus | 实例分割→bbox | WSI + 裁剪补集 | 1k² PNG/Npy | 1 606 / 188 图像；2 301 / 241 bbox | 0.5 | 248 |
| panuke | cell | 实例分割→bbox | 7 901 张 256² tile (~40 GB) | 256² NPY | 6 796 / 755 图像；164 626 / 18 396 bbox | 0.25 | 7 551 |
| puma | nucleus | GeoJSON polygon | 206 ROI (~16 GB) | 1024² PNG | 3 145 / 340 图像；104 816 / 10 637 bbox | 0.25 | 3 485 |
| segpc | cell | 细胞+核叠加 mask | 775 张 2560×1920 BMP | 2560×1920 BMP | 5 949 / 627 图像；13 539 / 1 374 bbox | 0.03125 | 6 576 |
| nuclick | nucleus | 实例分割→bbox | 1 463 张血液切片 (~452 MB) | 256² PNG | 5 927 / 663 图像；20 499 / 2 321 bbox | 0.25 | 6 590 |

> `outputs/patches` tile 数为 2025-11-06 统一导出的实际 train+val PNG 数量；若重新切片，请同步更新。

采样验证文件（示例，已备份至 `visuals/dataset_samples/`）：`breast_nucls/TCGA-A1-A0SP-...png`、`lizard/consep_1.png`、`conic/conic_sample_00000.png`、`monusac/MoNuSAC_TCGA-55-1594_001.png`、`panuke/panuke_part1_tile0.png`、`puma/training_set_metastatic_roi_001_poly.png`、`segpc/segpc_test_101.png`、`nuclick/nuclick_ROI_100_1.png`。MIDOG21/22 的 raw/tile 对照截图另存于 `visuals/raw_vs_tile_20251116/` 仅作记录。

## 2. 标签与 `CellAttributesV2` 的映射现状

| 数据集 | 原始标签 | 保留/丢弃策略 | 对应 `CellAttributesV2` 目标 | 备注 |
| --- | --- | --- | --- | --- |
| breast_nucls | tumor, fibroblast, lymphocyte, plasma cell, macrophage, mitotic figure, vascular endothelium, myoepithelium, apoptotic body, neutrophil, ductal epithelium, eosinophil | 全部保留；已在 `mappings.csv` 做手动对齐 | 侵袭性癌细胞/上皮细胞、Cancer-Associated Fibroblasts、Lymphocyte、Plasma cell、Macrophage、Mitosis (mitotic cell)、Endothelial cell、Myoepithelial cell、Apoptosis (apoptotic cell)、Neutrophil、Eosinophil | tumor/ductal 合并到"Epithelial cell"；fibroblast->CAF；myoepithelial 单独映射 |
| lizard | neutrophil, epithelial, lymphocyte, plasma, eosinophil, connective | 全部保留；connective→CAF | 对应各自细胞类型 | 读者脚本直接读取 `.mat`，映射已写入 `mappings.csv` |
| conic | 实例 id 1-6 | 全部保留；connective→CAF | 同上 | 读取 NPY tensors，映射已落盘 |
| monusac | epithelial, lymphocyte, macrophage, neutrophil, ambiguous | 丢弃 ambiguous | 其余四类直接对齐 | fallback 读取 `monusac_cropped/*.npy`，映射完备 |
| panuke | Neoplastic, Inflammatory, Connective/Soft tissue cells, Dead Cells, Epithelial | 丢弃 Background | Invasive cancer cells、Leukocyte (general)、Cancer-Associated Fibroblasts、Dead cells、Epithelial cell | 转 bbox 时已加入 CAF / Dead cells 描述 |
| puma | nuclei_* 十类 | stroma/histiocyte/melanophage → Macrophage | Invasive cancer cells、Lymphocyte、Plasma cell、Macrophage、Endothelial cell、Neutrophil、Epithelial cell、Apoptosis | `mappings.csv` 记录合并逻辑 |
| segpc | 单类 myeloma plasma cell | 全保留 | Plasma cell | `bone_segpc` MPP 已修正为 0.03125 |
| nuclick | 所有实例标记为白细胞 | 全保留 | Leukocyte (general) | `mappings.csv` 已补行；若要细分需额外特征 |

尚未纳入本轮训练但在计划中的数据集：TIGER ROI（标签"lymphocytes and plasma cells"需拆分）、SegPath（8 个免疫亚型，体量 570 GB）、GLySAC、DigestPath19、IHC-tlymph（权限待恢复）。这些数据的标签策略和 MPP 尚未落地。

## 3. 当前数据的潜在训练风险（更新）

- **尺度/面积差异依旧明显**：`docs/datasets/path_sam_bbox_stats.json`（2025-11-06）显示 MIDOG21/22、MoNuSAC、SEGPC 的平均 bbox 面积相对图像面积 >0.35，而 CoNIC/Lizard 仍在 0.007–0.014 区间；训练时仍需多尺度/anchor 调参或按数据集独立 loss 归一化。
- **tile 数失衡**：`outputs/patches` 统计显示 ConIC 23,617 张 tile、Lizard 8,494 张，而 SEGPC 仅 6,576 张（且多为大尺寸来源），batch 若平均采样仍会被前者主导；需通过 dataset-level reweight 或强制 oversample SEGPC/MoNuSAC 等小规模数据。
- **高分辨率下插值影响**：SEGPC（0.03125 µm/px）缩放到 0.25 需要 8× 下采样，虽避免 padding，但 bbox 边界需重点抽检（`outputs/patches/segpc_*` 已更新，可据此做 spot check）。
- **原始权限/数据完整性**：IHC t-lymph、SegPath 仍未纳入（权限问题）；若后续扩展需提前申请或复制原始目录。
- **训练任务尚在进行**：统一数据虽然就绪，但新 checkpoint 尚未产出；需在作业 368485 完成后重新评估 AP，确认归一化是否带来收益/风险。

## 4. 下一步行动

1. **监控训练**：跟踪 `logs/path_sam_ft_368485.out`，记录收敛曲线与最佳 checkpoint；完成后运行 `tools/eval_path_sam_all.py` 生成 per-dataset AP。
2. **采样/权重策略**：基于上表的 tile 估算，更新 `config/datasets_path_sam_full.json` 或采样器权重，确保 ConIC/Lizard 不完全主导 batch，SEGPC 等也不会完全稀释。
3. **文档联动**：将最新映射和 MPP 说明同步到 `docs/datasets/path_sam_odvg_plan.md`、`docs/datasets/path_sam_dataset_stats.json`，并在 `docs/experiments/timeline.md` 登记 2025-11-06 的全量重导出 + 训练启动。
4. **质检**：针对高缩放倍率数据（SEGPC、NuClick），抽查 `outputs/patches/*` 与原始图像，确认 0.25 MPP 下 bbox 仍合理；必要时在 `tools/build_cell_grounding.py` 中加入最小尺寸裁剪策略。
5. **扩展集准备**：若后续加入 SegPath/TIGER/GLySAC，沿用同一 reader/export 模板，提前在 `path_sam_mpp.json` 中补登记实际 MPP。

---- 完 ----
