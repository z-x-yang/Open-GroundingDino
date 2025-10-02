|                      | Cell/ Nucleus         | Cell Type                                                                                                                                                              | Protein marker |
| -------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| blood_nuclick        | cell mask             | white blood cell                                                                                                                                                       |                |
| bone_segpc           | cell + nucleus mask   | Myeloma Plasma Cell                                                                                                                                                    |                |
| breast_bcss          | cell mask             | plasma_cells<br />(lymphocytic_infiltrate, other_immune_infiltrate ToCheck)                                                                                            |                |
| breast_cahad         | cell/nuclei centroid  | mitosis, apoptosis, tumor nuclei, non-tumor nuclei, tubule, and non-tubule                                                                                             |                |
| breast_midog21       | cell bbox             | mitoses                                                                                                                                                                |                |
| breast_nucls         | nucleus bbox          | tumor, fibroblast, lymphocyte, plasma cell, macrophage, mitotic figure, vascular endothelium, myoepithelium, apoptotic body, neutrophil, ductal epithelium, eosinophil |                |
| breast_panoptils     | nucleus mask         | Cancer, stormal, large stromal, lympocyte, Plasma cell / large TIL nucleus, Normal epithelial, other, ambigious                                                      |                |
| breast_tiger         | cell mask             | Lymphocytes and plasma cells                                                                                                                                           |                |
| breast_tupac         | cell centroid         | mitoses                                                                                                                                                                |                |
| colon_conic          | nucleus mask          | neutrophil,epithelial, lymphocyte,plasma, eosinophil, connective                                                                                                    |                |
| colon_consep         | nucleus mask          | inflammatory, healthy epithelial, dysplastic/malignant epithelial, fibroblast, muscle, endothelial, other                                                        |                |
| colon_huncrc         |                       | tumor_necrosis To Check                                                                                                                                                |                |
| colon_lizard         | nucleus bbox+centroid | Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective tissue                                                                                         |                |
| gastric_digestpath19 | cell mask             | Signet_ring_cell                                                                                                                                                       |                |
| gastric_glysac       | nucleus mask          | TODO                                                                                                                                                                   |                |
| [ihc] endonuke       | nucleus mask          | stroma, epithelium, other nuclei                                                                                                                                      |                |
| [ihc] nuclick        | cell mask             | lymphocyte                                                                                                                                                             |                |
| [ihc] tlymphoctye    | cell bbox             | Other cell, Diverse, CD3+ immune cell, Tumor cell                                                                                                                     | Yes            |
| kidney_ocelot        | cell centroid        | Tumor Cell                                                                                                                                                             |                |
| mix_midog22_b        | cell bbox             | mitoses                                                                                                                                                                |                |
| mix_monusac20        | nuclear mask          | Epithelial, Macrophage, Neutrophil, Lymphocyte, Ambiguous                                                                                                          |                |
| mix_panuke           | nuclei mask          | Neoplastic, Inflammatory, Connective/Soft tissue cells, Dead Cells, Epithelial                                                                                     |                |
| mix_segpath          |                       | aSMA_SmoothMuscle, CD235a_RBC, CD3CD20_Lymphocyte, CD45RB_Leukocyte, ERG_Endothelium, MIST1_PlasmaCell, MNDA_MyeloidCell, panCK_Epithelium                            | Yes            |
| skin_adipoctye       | centroid              | adipoctye                                                                                                                                                              |                |
| skin_puma            | nuclei mask          | tumor, apoptosis, lymphocyte, endothelium, plasma_cell, stroma, histiocyte, melanophage, neutrophil, epithelium                                                        |                |

**Unknow type mask [cell/nuclues - background]**

* ING

|                                     | Cell/ Nucleus                   |
| ----------------------------------- | ------------------------------- |
| mix_cellseg                         | Cell                            |
| mix_hover                           | Cell                            |
| headneck_cytonuke                   | Cell + Nucleus                  |
| [cytology] cervic_cnsegnucleus mask | Nucleus Mask (TODO: #pts=1,3,4) |
| mix_CryoNuSeg                       | Nucleus                         |

**Other Stains Cell segmentation**

* https://www.kaggle.com/datasets/marinaeplissiti/sipakmed
* https://ieeexplore.ieee.org/abstract/document/10571965/references#references

**HE-IHC paired Image**

|             | #pair |
| ----------- | ----- |
| he-ihc_mist | 21295 |
| he-ihc_bci  | 4188  |

**Spatial Protein Expression**

* Coming soon
