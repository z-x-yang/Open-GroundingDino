# Path-SAM 数据集细胞类型分析

## 细胞类型分类标准

**细胞类型**: 指具有细胞核和细胞质的完整细胞实体
**非细胞类型**: 组织区域、血管、背景、分泌物等

## 各数据集细胞类型详细分析

### 🧬 Cell Segmentation（细胞分割）数据集

#### 1. blood_nuclick
- **细胞类型**: 白细胞 (white blood cell)
- **脚本**: `preprocess_seg/blood_nuclick.py`
- **说明**: 专门标注白细胞细胞的完整细胞边界

#### 2. breast_bcss  
- **细胞类型**: 浆细胞 (plasma_cells)
- **非细胞类型**: 肿瘤、基质、淋巴细胞浸润、坏死、腺体分泌物、血液、淋巴管、神经、皮肤附属器、血管、血管侵犯、DCIS等
- **脚本**: `preprocess_seg/breast_bcss.py`
- **说明**: 主要标注组织区域，浆细胞是唯一明确的细胞类型

#### 3. breast_tiger
- **细胞类型**: 淋巴细胞 (lymphocytes), 浆细胞 (plasma cells)
- **脚本**: `preprocess_seg/breast_tiger.py`
- **说明**: 专门标注淋巴细胞和浆细胞的细胞边界

#### 4. gastric_digestpath19
- **细胞类型**: 印戒细胞 (Signet_ring_cell)
- **脚本**: `preprocess_seg/gastric_digestpath19.py`
- **说明**: 专门标注胃癌中的印戒细胞

#### 5. ihc_nuclick
- **细胞类型**: 淋巴细胞 (lymphocyte)
- **脚本**: `preprocess_seg/ihc_nuclick.py`
- **说明**: IHC染色图像中的淋巴细胞细胞分割

#### 6. mix_cellseg
- **细胞类型**: 通用细胞 (generic cells)
- **脚本**: `preprocess_unknowntype_seg/mix_cellseg_bm.py`
- **说明**: 二值细胞分割，不区分具体细胞类型

#### 7. mix_hover
- **细胞类型**: 通用细胞 (generic cells)
- **脚本**: `preprocess_unknowntype_seg/mix_hover_bm.py`
- **说明**: 二值细胞分割，不区分具体细胞类型

### 🧪 Nucleus Mask（细胞核掩码）数据集

#### 1. bone_segpc
- **细胞类型**: 骨髓瘤浆细胞 (Myeloma Plasma Cell)
- **脚本**: `preprocess_seg/bone_segpc.py`
- **说明**: 包含细胞质和细胞核的完整细胞分割

#### 2. breast_panoptils
- **细胞类型**: 癌细胞核 (Cancer nucleus), 淋巴细胞核 (Lymphocyte nucleus), 浆细胞核 (Plasma cell nucleus), 正常上皮细胞核 (Normal epithelial nucleus)
- **非细胞类型**: 基质核 (Stromal nucleus), 大基质核 (Large stromal nucleus)
- **脚本**: `preprocess_seg/breast_panoptils.py`
- **说明**: 主要标注细胞核类型

#### 3. colon_conic
- **细胞类型**: 中性粒细胞 (neutrophil), 上皮细胞 (epithelial), 淋巴细胞 (lymphocyte), 浆细胞 (plasma), 嗜酸性粒细胞 (eosinophil)
- **非细胞类型**: 结缔组织 (connective)
- **脚本**: `preprocess_seg/colon_conic.py`
- **说明**: 结肠炎中的各种白细胞类型

#### 4. colon_consep
- **细胞类型**: 健康上皮细胞 (healthy epithelial), 发育不良/恶性上皮细胞 (dysplastic/malignant epithelial)
- **非细胞类型**: 炎症 (inflammatory), 成纤维细胞 (fibroblast), 肌肉 (muscle), 内皮细胞 (endothelial), 其他 (other)
- **脚本**: `preprocess_seg/colon_consep.py`
- **说明**: 结肠癌中的细胞核类型

#### 5. gastric_glysac
- **细胞类型**: 胃癌细胞核 (gastric cancer nuclei)
- **脚本**: `preprocess_seg/gastric_glysac.py`
- **说明**: 胃癌中的细胞核分割

#### 6. ihc_endonuke
- **细胞类型**: 上皮细胞核 (epithelium nuclei)
- **非细胞类型**: 基质核 (stroma nuclei), 其他细胞核 (other nuclei)
- **脚本**: `preprocess_seg/ihc_endonuke_c.py`
- **说明**: IHC染色中的细胞核类型

#### 7. mix_monusac20
- **细胞类型**: 上皮细胞 (Epithelial), 巨噬细胞 (Macrophage), 中性粒细胞 (Neutrophil), 淋巴细胞 (Lymphocyte)
- **非细胞类型**: 模糊 (Ambiguous)
- **脚本**: `preprocess_seg/mix_monusac20.py`
- **说明**: 混合器官中的白细胞类型

#### 8. mix_panuke
- **细胞类型**: 肿瘤细胞 (Neoplastic cells), 炎症细胞 (Inflammatory), 结缔组织细胞 (Connective/Soft tissue cells), 死细胞 (Dead Cells), 上皮细胞 (Epithelial)
- **脚本**: `preprocess_seg/mix_panuke.py`
- **说明**: 混合器官中的各种细胞类型

#### 9. skin_puma
- **细胞类型**: 肿瘤细胞 (tumor), 淋巴细胞 (lymphocyte), 浆细胞 (plasma_cell), 内皮细胞 (endothelium), 中性粒细胞 (neutrophil), 上皮细胞 (epithelium)
- **非细胞类型**: 凋亡 (apoptosis), 基质 (stroma), 组织细胞 (histiocyte), 黑色素吞噬细胞 (melanophage)
- **脚本**: `preprocess_seg/skin_puma.py`
- **说明**: 皮肤癌中的细胞核类型

#### 10. cervic_cnseg
- **细胞类型**: 宫颈癌细胞核 (cervical cancer nuclei)
- **脚本**: `preprocess_unknowntype_seg/cervic_cnseg_bm.py`
- **说明**: 宫颈癌中的细胞核分割

#### 11. mix_CryoNuSeg
- **细胞类型**: 混合细胞核 (mixed nuclei)
- **脚本**: `preprocess_unknowntype_seg/mix_CryoNuSeg_bm.py`
- **说明**: 冷冻切片中的细胞核分割

### 📦 Bounding Box（边界框）数据集

#### 1. breast_nucls
- **细胞类型**: 肿瘤细胞 (tumor), 成纤维细胞 (fibroblast), 淋巴细胞 (lymphocyte), 浆细胞 (plasma cell), 巨噬细胞 (macrophage), 有丝分裂细胞 (mitotic figure), 血管内皮细胞 (vascular endothelium), 肌上皮细胞 (myoepithelium), 凋亡细胞 (apoptotic body), 中性粒细胞 (neutrophil), 导管上皮细胞 (ductal epithelium), 嗜酸性粒细胞 (eosinophil)
- **脚本**: `preprocess_seg/breast_nucls_b.py`
- **说明**: 乳腺癌中的12种细胞核类型

#### 2. breast_midog21
- **细胞类型**: 有丝分裂细胞 (mitoses)
- **脚本**: `preprocess_seg/breast_midog21_b.py`
- **说明**: 乳腺癌中的有丝分裂细胞检测

#### 3. ihc_tlymphoctype
- **细胞类型**: CD3+免疫细胞 (CD3+ immune cell), 肿瘤细胞 (Tumor cell)
- **非细胞类型**: 其他细胞 (Other cell), 多样细胞 (Diverse)
- **脚本**: `preprocess_seg/ihc_tlymphoctype_b.py`
- **说明**: IHC染色中的T淋巴细胞分类

#### 4. mix_midog22_b
- **细胞类型**: 有丝分裂细胞 (mitoses)
- **脚本**: `preprocess_seg/mix_midog22_b.py`
- **说明**: 混合器官中的有丝分裂细胞检测

### 📍 Centroid（质心点）数据集

#### 1. breast_cahad
- **细胞类型**: 有丝分裂细胞 (mitosis), 凋亡细胞 (apoptosis), 肿瘤细胞核 (tumor nuclei), 非肿瘤细胞核 (non-tumor nuclei)
- **非细胞类型**: 管状结构 (tubule), 非管状结构 (non-tubule)
- **脚本**: `preprocess_seg/breast_cahad_c.py`
- **说明**: 乳腺癌中的细胞质心点标注

#### 2. breast_tupac
- **细胞类型**: 有丝分裂细胞 (mitoses)
- **脚本**: `preprocess_seg/breast_tupac_c.py`
- **说明**: 乳腺癌中的有丝分裂细胞质心点

#### 3. kidney_ocelot
- **细胞类型**: 肿瘤细胞 (Tumor Cell)
- **脚本**: `preprocess_seg/kidney_ocelot_c.py`
- **说明**: 肾脏中的肿瘤细胞质心点

#### 4. skin_adipocyte
- **细胞类型**: 脂肪细胞 (adipocyte)
- **脚本**: `preprocess_seg/skin_adipocyte.py`
- **说明**: 皮肤中的脂肪细胞质心点

### 🔬 混合标注类型数据集

#### 1. headneck_cytonuke
- **细胞类型**: 细胞 (CELL), 细胞核 (NUCLEUS)
- **脚本**: `preprocess_unknowntype_seg/headneck_cytonuke_bm.py`
- **说明**: 头颈癌中的细胞和细胞核实例分割

#### 2. colon_lizard
- **细胞类型**: 中性粒细胞 (Neutrophil), 上皮细胞 (Epithelial), 淋巴细胞 (Lymphocyte), 浆细胞 (Plasma), 嗜酸性粒细胞 (Eosinophil)
- **非细胞类型**: 结缔组织 (Connective tissue)
- **脚本**: `preprocess_seg/colon_lizard_b_c.py`
- **说明**: 结肠中的细胞核边界框和质心点

### 🧪 蛋白质标记数据集

#### 1. mix_segpath
- **细胞类型**: 淋巴细胞 (Lymphocyte), 白细胞 (Leukocyte), 内皮细胞 (Endothelium), 浆细胞 (PlasmaCell), 髓样细胞 (MyeloidCell), 上皮细胞 (Epithelium)
- **非细胞类型**: 平滑肌 (SmoothMuscle), 红细胞 (RBC)
- **脚本**: `preprocess_seg/mix_segpath.py`
- **说明**: 基于蛋白质标记的细胞类型分类

## 总结

**纯细胞类型数据集** (只包含细胞类型):
- blood_nuclick, breast_tiger, gastric_digestpath19, ihc_nuclick, mix_cellseg, mix_hover
- bone_segpc, gastric_glysac, cervic_cnseg, mix_CryoNuSeg
- breast_midog21, breast_tupac, kidney_ocelot, skin_adipocyte, mix_midog22_b
- headneck_cytonuke

**混合类型数据集** (包含细胞和非细胞类型):
- breast_bcss, breast_panoptils, colon_conic, colon_consep, ihc_endonuke, mix_monusac20, mix_panuke, skin_puma
- breast_nucls, ihc_tlymphoctype, breast_cahad, colon_lizard, mix_segpath
