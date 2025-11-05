import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

import datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from models import build_model as _build_model


def load_model(cfg_path: str, ckpt_path: str, device: str):
    args = SLConfig.fromfile(cfg_path)
    args.use_coco_eval = False
    args.coco_val_path = getattr(args, "coco_val_path", "")
    args.label_list = getattr(args, "label_list", [])
    args.device = device
    model, _, _ = _build_model(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()
    return model


def transform_image(image_path: str):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def run_inference(model, image, prompt: str, box_threshold: float, text_threshold: float, device: str):
    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."
    with torch.no_grad():
        outputs = model(image.unsqueeze(0).to(device), captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[mask]
    boxes_filt = boxes[mask]
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    phrases = []
    for logit in logits_filt:
        token_mask = logit > text_threshold
        tokens = [tokenized["input_ids"][i] for i, m in enumerate(token_mask) if m]
        if tokens:
            phrases.append(tokenizer.decode(tokens))
        else:
            phrases.append("")
    return boxes_filt.cpu(), phrases


def draw_boxes(image_pil, boxes, phrases):
    W, H = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()
    for box, phrase in zip(boxes, phrases):
        x_c, y_c, w, h = box
        x0 = (x_c - w / 2) * W
        y0 = (y_c - h / 2) * H
        x1 = (x_c + w / 2) * W
        y1 = (y_c + h / 2) * H
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0), phrase[:60], fill="white", font=font)
    return image_pil


def main():
    parser = argparse.ArgumentParser(description="Visualize cell/nucleus detections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--box_thresh", type=float, default=0.3)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = load_model(args.config, args.ckpt, device)
    image_pil, image = transform_image(args.image)
    boxes, phrases = run_inference(model, image, args.prompt, args.box_thresh, args.text_thresh, device)
    vis = draw_boxes(image_pil, boxes, phrases)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    vis.save(args.output)


if __name__ == "__main__":
    main()
