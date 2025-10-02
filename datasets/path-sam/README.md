# Path-SAM 医学图像数据集预处理脚本

本文件夹包含了27个医学图像数据集的预处理脚本，涵盖了细胞分割、细胞核分割、边界框检测等多种标注类型。

## 📁 目录结构

```
datasets/path-sam/
├── README.md                           # 本文件
├── utils/                              # 工具函数
│   ├── cfg.py                         # 数据路径配置
│   └── draw.py                        # 可视化工具
├── preprocess/                         # 通用预处理工具
│   ├── segment_utils.py               # 组织分割工具
│   └── tile_utils.py                  # 图像分块工具
├── preprocess_seg/                     # 分割数据集预处理脚本
├── preprocess_unknowntype_seg/         # 未知类型分割数据集脚本
└── he-ihc_pairs/                       # HE-IHC配对数据集脚本
```

## 🔧 数据路径配置

所有脚本使用统一的路径配置（`utils/cfg.py`）：
- **原始数据路径**: `VLDATA_RAW = '/n/lw_groups/hms/dbmi/yu/lab/seg_data_raw'`
- **处理后数据路径**: `VLDATA_PROCESS = '/n/lw_groups/hms/dbmi/yu/lab/xug751/datavl/seg_data'`

## 📊 数据集分类

### 🧬 Cell Segmentation（细胞分割）数据集

| 数据集 | 脚本文件 | 原始数据位置 | 处理后数据位置 | 标注类型 | 细胞类型 |
|--------|----------|-------------|---------------|----------|----------|
| **blood_nuclick** | `preprocess_seg/blood_nuclick.py` | `{VLDATA_RAW}/nuclick` | `{VLDATA_PROCESS}/blood_nuclick.json` | Cell Mask | 白细胞 |
| **breast_bcss** | `preprocess_seg/breast_bcss.py` | `{VLDATA_RAW}/BCSS/BCSS` | `{VLDATA_PROCESS}/breast_bcss.json` | Cell Mask | 肿瘤、基质、淋巴细胞浸润等 |
| **breast_tiger** | `preprocess_seg/breast_tiger.py` | `{VLDATA_RAW}/tiger` | `{VLDATA_PROCESS}/breast_tiger.json` | Cell Mask | 淋巴细胞和浆细胞 |
| **gastric_digestpath19** | `preprocess_seg/gastric_digestpath19.py` | `{VLDATA_RAW}/digestpath19` | `{VLDATA_PROCESS}/gastric_digestpath19.json` | Cell Mask | 印戒细胞 |
| **ihc_nuclick** | `preprocess_seg/ihc_nuclick.py` | `{VLDATA_RAW}/ihc_nuclick` | `{VLDATA_PROCESS}/ihc_nuclick.json` | Cell Mask | 淋巴细胞（IHC染色） |
| **mix_cellseg** | `preprocess_unknowntype_seg/mix_cellseg_bm.py` | `{VLDATA_RAW}/cellseg` | `{VLDATA_PROCESS}/mix_cellseg.json` | Cell Mask | 未知类型细胞 |
| **mix_hover** | `preprocess_unknowntype_seg/mix_hover_bm.py` | `{VLDATA_RAW}/hover` | `{VLDATA_PROCESS}/mix_hover.json` | Cell Mask | 未知类型细胞 |

### 🧪 Nucleus Mask（细胞核掩码）数据集

| 数据集 | 脚本文件 | 原始数据位置 | 处理后数据位置 | 标注类型 | 细胞核类型 |
|--------|----------|-------------|---------------|----------|------------|
| **bone_segpc** | `preprocess_seg/bone_segpc.py` | `{VLDATA_RAW}/segpc` | `{VLDATA_PROCESS}/bone_segpc.json` | Cell + Nucleus Mask | 骨髓瘤浆细胞 |
| **breast_panoptils** | `preprocess_seg/breast_panoptils.py` | `{VLDATA_RAW}/panoptils` | `{VLDATA_PROCESS}/breast_panoptils.json` | Nucleus Mask | 癌症、基质、淋巴细胞等 |
| **colon_conic** | `preprocess_seg/colon_conic.py` | `{VLDATA_RAW}/CoNIC/data` | `{VLDATA_PROCESS}/colon_conic.json` | Nucleus Mask | 中性粒细胞、上皮细胞、淋巴细胞等 |
| **colon_consep** | `preprocess_seg/colon_consep.py` | `{VLDATA_RAW}/consep` | `{VLDATA_PROCESS}/colon_consep.json` | Nucleus Mask | 炎症、健康上皮、恶性上皮等 |
| **gastric_glysac** | `preprocess_seg/gastric_glysac.py` | `{VLDATA_RAW}/glysac` | `{VLDATA_PROCESS}/gastric_glysac.json` | Nucleus Mask | 胃癌细胞核 |
| **ihc_endonuke** | `preprocess_seg/ihc_endonuke_c.py` | `{VLDATA_RAW}/endonuke` | `{VLDATA_PROCESS}/ihc_endonuke.json` | Nucleus Mask | 基质、上皮、其他细胞核 |
| **mix_monusac20** | `preprocess_seg/mix_monusac20.py` | `{VLDATA_RAW}/monusac20` | `{VLDATA_PROCESS}/mix_monusac20.json` | Nucleus Mask | 上皮、巨噬细胞、中性粒细胞等 |
| **mix_panuke** | `preprocess_seg/mix_panuke.py` | `{VLDATA_RAW}/panuke` | `{VLDATA_PROCESS}/mix_panuke.json` | Nucleus Mask | 肿瘤性、炎症性、结缔组织等 |
| **skin_puma** | `preprocess_seg/skin_puma.py` | `{VLDATA_RAW}/puma` | `{VLDATA_PROCESS}/skin_puma.json` | Nucleus Mask | 肿瘤、凋亡、淋巴细胞等 |
| **cervic_cnseg** | `preprocess_unknowntype_seg/cervic_cnseg_bm.py` | `{VLDATA_RAW}/cnseg` | `{VLDATA_PROCESS}/cervic_cnseg.json` | Nucleus Mask | 宫颈癌细胞核 |
| **mix_CryoNuSeg** | `preprocess_unknowntype_seg/mix_CryoNuSeg_bm.py` | `{VLDATA_RAW}/cryonuseg` | `{VLDATA_PROCESS}/mix_CryoNuSeg.json` | Nucleus Mask | 混合细胞核 |

### 📦 Bounding Box（边界框）数据集

| 数据集 | 脚本文件 | 原始数据位置 | 处理后数据位置 | 标注类型 | 目标类型 |
|--------|----------|-------------|---------------|----------|----------|
| **breast_nucls** | `preprocess_seg/breast_nucls_b.py` | `{VLDATA_RAW}/nucls` | `{VLDATA_PROCESS}/breast_nucls.json` | Nucleus BBox | 12种细胞核类型 |
| **breast_midog21** | `preprocess_seg/breast_midog21_b.py` | `{VLDATA_RAW}/MIDOG21` | `{VLDATA_PROCESS}/breast_midog21.json` | Cell BBox | 有丝分裂 |
| **ihc_tlymphoctype** | `preprocess_seg/ihc_tlymphoctype_b.py` | `{VLDATA_RAW}/tlymphoctype` | `{VLDATA_PROCESS}/ihc_tlymphoctype.json` | Cell BBox | T淋巴细胞（含蛋白质标记） |
| **mix_midog22_b** | `preprocess_seg/mix_midog22_b.py` | `{VLDATA_RAW}/midog2` | `{VLDATA_PROCESS}/mix_midog22.json` | Cell BBox | 有丝分裂 |

### 📍 Centroid（质心点）数据集

| 数据集 | 脚本文件 | 原始数据位置 | 处理后数据位置 | 标注类型 | 目标类型 |
|--------|----------|-------------|---------------|----------|----------|
| **breast_cahad** | `preprocess_seg/breast_cahad_c.py` | `{VLDATA_RAW}/BreCaHAD` | `{VLDATA_RAW}/BreCaHAD_processed` | Cell/Nuclei Centroid | 有丝分裂、凋亡、肿瘤等 |
| **breast_tupac** | `preprocess_seg/breast_tupac_c.py` | `{VLDATA_RAW}/TUPAC-mitoses` | `{VLDATA_RAW}/TUPAC-mitoses/process_patch` | Cell Centroid | 有丝分裂 |
| **kidney_ocelot** | `preprocess_seg/kidney_ocelot_c.py` | `{VLDATA_RAW}/ocelot_processed` | `{VLDATA_RAW}/ocelot_processed/test_patches` | Cell Centroid | 肿瘤细胞 |
| **skin_adipocyte** | `preprocess_seg/skin_adipocyte.py` | `{VLDATA_RAW}/adipocyte` | `{VLDATA_PROCESS}/skin_adipocyte.json` | Centroid | 脂肪细胞 |

### 🔬 混合标注类型数据集

| 数据集 | 脚本文件 | 原始数据位置 | 处理后数据位置 | 标注类型 | 目标类型 |
|--------|----------|-------------|---------------|----------|----------|
| **headneck_cytonuke** | `preprocess_unknowntype_seg/headneck_cytonuke_bm.py` | `{VLDATA_RAW}/cytonuke` | `{VLDATA_PROCESS}/headneck_cytonuke.json` | Cell + Nucleus | 头颈癌细胞 |
| **colon_lizard** | `preprocess_seg/colon_lizard_b_c.py` | `{VLDATA_RAW}/Lizard` | `{VLDATA_PROCESS}/colon_lizard.json` | Nucleus BBox + Centroid | 6种细胞核类型 |
| **colon_huncrc** | `preprocess_seg/colon_huncrc.py` | `{VLDATA_RAW}/HunCRC` | `{VLDATA_PROCESS}/colon_huncrc.json` | WSI Region | 肿瘤坏死等 |

### 🎨 HE-IHC配对数据集

| 数据集 | 脚本文件 | 原始数据位置 | 处理后数据位置 | 图像对数 | 标记类型 |
|--------|----------|-------------|---------------|----------|----------|
| **he-ihc_mist** | `he-ihc_pairs/he-ihc_mist.py` | `{VLDATA_RAW}/mist` | - | 21,295对 | PR, ER, HER2, Ki67 |
| **he-ihc_bci** | `he-ihc_pairs/he-ihc_bci.py` | `{VLDATA_RAW}/BCI_dataset` | - | 4,188对 | HE-IHC配对 |

### 🧪 蛋白质标记数据集

| 数据集 | 脚本文件 | 原始数据位置 | 处理后数据位置 | 标注类型 | 蛋白质标记 |
|--------|----------|-------------|---------------|----------|------------|
| **mix_segpath** | `preprocess_seg/mix_segpath.py` | `{VLDATA_RAW}/segpath` | `{VLDATA_PROCESS}/mix_segpath.json` | Protein Expression | aSMA, CD235a, CD3CD20, CD45RB, ERG, MIST1, MNDA, panCK |

## 📋 数据格式说明

### 预处理前格式
- **图像格式**: PNG, TIFF, BMP, JPG
- **标注格式**: 
  - JSON文件（COCO格式、自定义格式）
  - CSV文件（质心坐标、边界框）
  - MAT文件（MATLAB格式）
  - 掩码图像（PNG）
  - WSI文件（MRXS）

### 预处理后格式
- **图像格式**: PNG（统一为RGB格式）
- **标注格式**: JSON文件，包含以下字段：
  ```json
  {
    "image": "图像文件路径",
    "label": {
      "细胞类型1": "像素比例",
      "细胞类型2": "像素比例"
    },
    "count": {
      "细胞类型1": "细胞数量",
      "细胞类型2": "细胞数量"
    }
  }
  ```

## 🛠️ 通用预处理工具

### `preprocess/tile_utils.py`
- **功能**: 图像分块、组织分割、标注转换
- **主要函数**:
  - `crop_img_mask_to_caption()`: 将掩码转换为描述文本
  - `crop_img_mask()`: 基于掩码的图像分块
  - `crop_img_wcountour()`: 基于轮廓的图像分块

### `preprocess/segment_utils.py`
- **功能**: 组织区域分割
- **用途**: 自动识别组织区域，去除背景

### `utils/draw.py`
- **功能**: 可视化工具
- **用途**: 绘制边界框、轮廓、掩码等

## 🚀 使用示例

### 运行单个数据集预处理
```bash
cd datasets/path-sam/preprocess_seg
python breast_nucls_b.py
```

### 批量处理多个数据集
```bash
cd datasets/path-sam/preprocess_seg
for script in *.py; do
    echo "Processing $script"
    python "$script"
done
```

## 📝 注意事项

1. **路径配置**: 确保 `utils/cfg.py` 中的路径配置正确
2. **依赖库**: 需要安装 OpenSlide、scipy、opencv-python 等库
3. **内存使用**: 某些WSI数据集需要大量内存，建议分批处理
4. **文件权限**: 确保对输出目录有写入权限
5. **数据完整性**: 运行前请检查原始数据是否完整

## 🔗 相关链接

- [GroundingDINO 项目](https://github.com/IDEA-Research/GroundingDINO)
- [Open GroundingDino](https://github.com/longzw1997/Open-GroundingDino)
- [Path-SAM 原始仓库](https://github.com/gong-xuan/path-sam)

---

**最后更新**: 2024年10月2日  
**维护者**: z-x-yang
