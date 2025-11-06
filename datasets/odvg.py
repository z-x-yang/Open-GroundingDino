import datasets.transforms as T
from datasets.coco import make_coco_transforms as _coco_make_transforms
from torchvision.datasets.vision import VisionDataset
import os.path
from typing import Callable, Optional
import json
from PIL import Image
import torch
import random
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))


class ODVGDataset(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        anno (string): Path to json annotation file.
        label_map_anno (string):  Path to json label mapping file. Only for Object Detection
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _attr_df = None
    _tokenizer = None
    _prompt_gen = None
    _xlsx_utils = None

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()
        self.dynamic_prompt = os.environ.get("ODVG_DYNAMIC_PROMPT", "1") != "0"

    def _lazy_init_dynamic(self):
        if not self.dynamic_prompt or self.dataset_mode != "VG":
            return
        if ODVGDataset._prompt_gen is None or ODVGDataset._xlsx_utils is None:
            # lazy import utilities
            repo_root = os.path.dirname(os.path.dirname(__file__))
            build_dir = os.path.join(
                repo_root, 'datasets', 'path-sam', 'build')
            sys.path.append(build_dir)
            try:
                from datasets.path_sam.build import prompt_generator as _pg  # type: ignore
            except Exception:
                import importlib.util
                pg_path = os.path.join(build_dir, 'prompt_generator.py')
                spec = importlib.util.spec_from_file_location(
                    'prompt_generator', pg_path)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                _pg = mod
            try:
                from datasets.path_sam.build import xlsx_utils as _xu  # type: ignore
            except Exception:
                import importlib.util
                xu_path = os.path.join(build_dir, 'xlsx_utils.py')
                spec = importlib.util.spec_from_file_location(
                    'xlsx_utils', xu_path)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                _xu = mod
            ODVGDataset._prompt_gen = _pg
            ODVGDataset._xlsx_utils = _xu
        if ODVGDataset._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                ODVGDataset._tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-uncased")
            except Exception:
                ODVGDataset._tokenizer = None
        if ODVGDataset._attr_df is None:
            # load xlsx
            csv_path = os.path.join(os.path.dirname(os.path.dirname(
                __file__)), 'datasets', 'path-sam', 'CellAttributesV2.csv')
            try:
                df, _ = ODVGDataset._xlsx_utils.load_cell_attributes_csv(csv_path)
                ODVGDataset._attr_df = df
            except Exception:
                ODVGDataset._attr_df = None

    def load_label_map(self, label_map_anno):
        with open(label_map_anno, 'r') as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        with open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        print(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            pass

    def __getitem__(self, index: int):
        meta = self.metas[index]
        rel_path = meta["filename"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found.")
        image = Image.open(abs_path).convert('RGB')
        w, h = image.size
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            # neg bbox labels
            neg_labels = list(self.label_index.difference(pos_labels))

            vg_labels = list(pos_labels)
            num_to_add = min(len(neg_labels), self.max_labels-len(pos_labels))
            if num_to_add > 0:
                vg_labels.extend(random.sample(neg_labels, num_to_add))

            # shuffle
            for i in range(len(vg_labels)-1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [self.label_map[lb] for lb in vg_labels]
            caption_dict = {item: index for index,
                            item in enumerate(caption_list)}

            caption = ' . '.join(caption_list) + ' .'
            classes = [
                caption_dict[self.label_map[str(obj["label"])]] for obj in instances]

            classes = torch.tensor(classes, dtype=torch.int64)
        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            if self.dynamic_prompt:
                self._lazy_init_dynamic()
                if ODVGDataset._attr_df is not None and ODVGDataset._tokenizer is not None and ODVGDataset._prompt_gen is not None:
                    new_phrases = []
                    for phr in caption_list:
                        obj_type, std = self._parse_object_and_type(phr)
                        if obj_type and std:
                            try:
                                dyn = ODVGDataset._prompt_gen.gen_unique_description_dynamic(
                                    obj_type, std, ODVGDataset._attr_df, ODVGDataset._tokenizer, max_tokens=256)
                                new_phrases.append(dyn)
                            except Exception:
                                new_phrases.append(phr)
                        else:
                            new_phrases.append(phr)
                    caption_list = new_phrases
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            # preserve order of first occurrence to keep class indices aligned with caption tokens
            seen = set()
            uni_caption_list = []
            for cap in caption_list:
                if cap not in seen:
                    seen.add(cap)
                    uni_caption_list.append(cap)
            # enforce total caption token length ≤ max_text_len (256) by dropping tail phrases
            max_len = 256
            try:
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception:
                _tok = None
            if _tok is not None:
                kept = []
                for cap in uni_caption_list:
                    trial = (' . '.join(kept + [cap]) +
                             ' .') if kept else (cap + ' .')
                    if len(_tok(trial)["input_ids"]) <= max_len:
                        kept.append(cap)
                    else:
                        break
                uni_caption_list = kept if kept else uni_caption_list[:1]
                # filter classes to kept set only
                kept_set = set(uni_caption_list)
                caption_list = [cap for cap in caption_list if cap in kept_set]
                boxes = [b for b, cap in c if cap in kept_set]
            label_map = {cap: i for i, cap in enumerate(uni_caption_list)}
            classes = [label_map[cap] for cap in caption_list]
            caption = ' . '.join(uni_caption_list) + ' .'
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            classes = torch.tensor(classes, dtype=torch.int64)
            caption_list = uni_caption_list
        target = {}
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes
        # size, cap_list, caption, bboxes, labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _parse_object_and_type(self, phrase: str):
        try:
            p = phrase
            obj = None
            typ = None
            if 'object=' in p:
                s = p.split('object=', 1)[1]
                obj = s.split(';', 1)[0].strip().strip('.')
            if 'type=' in p:
                s = p.split('type=', 1)[1]
                typ = s.split(';', 1)[0].strip().strip('.')
            return obj, typ
        except Exception:
            return None, None

    def __len__(self) -> int:
        return len(self.metas)


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    # use the standard coco transforms from datasets.coco
    return _coco_make_transforms(image_set, fix_size=fix_size, strong_aug=strong_aug, args=args)


def build_odvg(image_set, args, datasetinfo):
    img_folder = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    label_map = datasetinfo["label_map"] if "label_map" in datasetinfo else None
    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    print(img_folder, ann_file, label_map)
    dataset = ODVGDataset(img_folder, ann_file, label_map, max_labels=args.max_labels,
                          transforms=make_coco_transforms(
                              image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args),
                          )
    return dataset


if __name__ == "__main__":
    dataset_vg = ODVGDataset("path/GRIT-20M/data/",
                             "path/GRIT-20M/anno/grit_odvg_10k.jsonl",)
    print(len(dataset_vg))
    data = dataset_vg[random.randint(0, 100)]
    print(data)
    dataset_od = ODVGDataset("pathl/V3Det/",
                             "path/V3Det/annotations/v3det_2023_v1_all_odvg.jsonl",
                             "path/V3Det/annotations/v3det_label_map.json",
                             )
    print(len(dataset_od))
    data = dataset_od[random.randint(0, 100)]
    print(data)
