# python infer.py --img data/test --ckpt outputs/fasterrcnn.pth --th 0.5

# infer.py ─────────────────────────────────────────────
import argparse, ast, pathlib, os, torch, torchvision
from torchvision.io import read_image, write_png
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PALETTE = {
    1: "red",  # Bus
    2: "green",  # Car
    3: "orange",  # Motorcycle
    4: "blue",  # Pickup
    5: "purple",  # Truck
}


# ---------- 1. 讀取 label map ----------
def load_label_map(txt_path="data/category.txt"):
    raw = pathlib.Path(txt_path).read_text().strip()
    if raw.lower().startswith("category"):
        raw = raw.split(":", 1)[1].strip()
    d = ast.literal_eval(raw)
    return {k: v + 1 for k, v in d.items()}


LABEL_MAP = load_label_map()
ID2NAME = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = max(ID2NAME) + 1  # 6


# ---------- 2. 構建模型並載入權重 ----------
def get_model(weight_path: str):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_ch = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_ch, NUM_CLASSES)
    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    return model


# ---------- 3. 推論單張影像 ----------
@torch.inference_mode()
def predict_one(model, img_tensor, score_th=0.5, device="cpu"):
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float() / 255.0
    pred = model([img_tensor.to(device)])[0]
    keep = pred["scores"] >= score_th
    boxes = pred["boxes"][keep].cpu().round().int()
    labels = pred["labels"][keep].cpu()
    scores = pred["scores"][keep].cpu()

    texts = [f"{ID2NAME[int(l)]:s}:{s:.2f}" for l, s in zip(labels, scores)]
    colors = [PALETTE[int(l)] for l in labels]
    drawn = draw_bounding_boxes(
        (img_tensor * 255).byte().cpu(), boxes, labels=texts, colors=colors, width=2
    )
    return drawn, scores


# ---------- 4. 主程式 ----------
def main():
    parser = argparse.ArgumentParser(description="Faster R-CNN inference script")
    parser.add_argument(
        "--img", type=str, required=True, help="單張圖片路徑或資料夾 (*.jpg *.png)"
    )
    parser.add_argument(
        "--ckpt", type=str, default="outputs/fasterrcnn.pth", help="模型權重檔 (.pth)"
    )
    parser.add_argument("--out", type=str, default="outputs", help="輸出資料夾")
    parser.add_argument(
        "--th", type=float, default=0.5, help="score 門檻 (default=0.5)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.ckpt).to(device)

    p = pathlib.Path(args.img)
    paths = (list(p.glob("*.jpg")) + list(p.glob("*.png"))) if p.is_dir() else [p]

    os.makedirs(args.out, exist_ok=True)

    score_sum, det_cnt = 0.0, 0

    for img_path in paths:
        img = read_image(str(img_path))
        drawn, scores = predict_one(model, img, args.th, device)

        # -- save —
        out_file = pathlib.Path(args.out) / f"{img_path.stem}_pred.png"
        write_png(drawn, str(out_file))
        print(f"{img_path.name:20s}  →  {out_file.name:20s}  ({len(scores)} objects)")

        score_sum += float(scores.sum())
        det_cnt += len(scores)

    if det_cnt:
        avg_score = score_sum / det_cnt
        print(
            f"\n🌟 Average score over ALL detections: {avg_score:.3f} "
            f"(門檻 {args.th})"
        )
    else:
        print("\n⚠️  No detections above threshold — average score = 0.0")


if __name__ == "__main__":
    main()
