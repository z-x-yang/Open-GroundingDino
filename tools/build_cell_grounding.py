import os
import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image

from datasets.path_sam.build.xlsx_utils import load_cell_attributes
from datasets.path_sam.build.mapping_builder import build_initial_mappings
from datasets.path_sam.build.patcher import tile_image, project_and_filter_bboxes
from datasets.path_sam.build.odvg_exporter import write_grounding_jsonl
from datasets.path_sam.build.coco_exporter import export_coco


def main():
    # Skeleton: wire up modules later with dataset readers
    print("[build] cell grounding pipeline skeleton ready")


if __name__ == "__main__":
    main()


