import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

import datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util import box_ops
from models import build_model as build_grounding


def load_model(cfg_path: str, ckpt_path: str, device: str):
    args = SLConfig.fromfile(cfg_path)
    # disable coco eval requirements for standalone inference
    args.use_coco_eval = False
    args.coco_val_path = getattr(args, "coco_val_path", "")
    args.label_list = getattr(args, "label_list", [])
    args.device = device
    model, _, _ = build_grounding(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()
    return model


def transform_image(image_path: Path):
    pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(pil, None)
    return pil, image


def run_inference(model, image_tensor: torch.Tensor, prompt: str, box_thr: float, text_thr: float, device: str):
    caption = prompt.lower().strip()
    if not caption.endswith('.'):
        caption += '.'
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device), captions=[caption])
    logits = outputs['pred_logits'].sigmoid()[0]
    boxes = outputs['pred_boxes'][0]
    scores = logits.max(dim=1)[0]
    mask = scores > box_thr
    logits = logits[mask]
    boxes = boxes[mask]
    scores = scores[mask]
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = []
    for logit in logits:
        token_mask = logit > text_thr
        token_ids = [tokenized['input_ids'][i] for i, keep in enumerate(token_mask) if keep]
        if token_ids:
            phrases.append(tokenizer.decode(token_ids))
        else:
            phrases.append('')
    return boxes.cpu(), phrases, scores.cpu()


def cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int):
    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    xyxy[:, [0, 2]] *= width
    xyxy[:, [1, 3]] *= height
    return xyxy


def draw_boxes(image: Image.Image, boxes: List[Tuple], labels: List[str], color: str):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, y0), label, fill=color, font=font)


def load_record(jsonl_path: Path, index: int):
    with jsonl_path.open() as f:
        for idx, line in enumerate(f):
            if idx == index:
                return json.loads(line)
    raise IndexError('Index out of range')


def main():
    parser = argparse.ArgumentParser(description='Compare predictions vs GT for one record')
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--box_thr', type=float, default=0.3)
    parser.add_argument('--text_thr', type=float, default=0.25)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    record = load_record(Path(args.jsonl), args.index)
    img_path = Path(record['filename'])
    pil, tensor = transform_image(img_path)
    width, height = pil.size

    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    model = load_model(args.config, args.ckpt, device)

    if args.prompt:
        prompt = args.prompt
        gt_boxes = []
        gt_labels = []
    else:
        regions = record['grounding']['regions']
        # use first region phrase as prompt
        prompt = regions[0]['phrase']
        gt_boxes = [r['bbox'] for r in regions]
        gt_labels = [r['phrase'] for r in regions]

    boxes, phrases, scores = run_inference(model, tensor, prompt, args.box_thr, args.text_thr, device)
    pred_boxes = cxcywh_to_xyxy(boxes, width, height)
    pred_labels = [f"{p[:25]}|{s:.2f}" for p, s in zip(phrases, scores)]

    vis = pil.copy()
    if gt_boxes:
        draw_boxes(vis, gt_boxes, ['GT'] * len(gt_boxes), 'green')
    draw_boxes(vis, pred_boxes.tolist(), pred_labels, 'red')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    vis.save(args.output)
    summary = {
        'image': str(img_path),
        'prompt': prompt,
        'gt_boxes': gt_boxes,
        'pred': [{'box': box.tolist(), 'phrase': phrase, 'score': float(score)} for box, phrase, score in zip(pred_boxes, phrases, scores)],
    }
    summary_path = Path(args.output).with_suffix('.json')
    summary_path.write_text(json.dumps(summary, indent=2))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
