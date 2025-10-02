# Label Mappings (for audit)

This document records the canonical mapping from each dataset's original labels to the unified cell/nucleus taxonomy aligned with `CellType2Attributes.xlsx` (sheet: CellAttributes).

Schema (see also `mappings.csv`):

- dataset: breast_nucls | breast_midog21 | ihc_tlymphoctype | mix_midog22_b
- original_label: original category name from the dataset
- object_type: cell | nucleus
- standard_cell_name: exact value of the "Cell Type" column in the xlsx (must match exactly)
- keep: true|false (if false, the instance will be dropped)
- note: free text, e.g., "sparse class", "needs manual confirmation"

Initial entries (to be expanded programmatically):

| dataset           | original_label         | object_type | standard_cell_name   | keep  | note |
|-------------------|------------------------|-------------|----------------------|-------|------|
| ihc_tlymphoctype  | CD3+ immune cell       | cell        | Lymphocyte           | true  |      |
| ihc_tlymphoctype  | Tumor cell             | cell        | Tumor cell           | true  |      |
| ihc_tlymphoctype  | Diverse                | cell        |                      | false | drop |
| ihc_tlymphoctype  | Other cell             | cell        |                      | false | drop |

Guidelines:

1) Only rows with keep=true will be used to construct training/validation annotations.
2) For breast_nucls, each nucleus category should be mapped to a nucleus object_type with the closest matching "Cell Type"; categories not present in the xlsx must be kept=false.
3) This file is the single source of truth for label decisions; any changes should be reviewed.


