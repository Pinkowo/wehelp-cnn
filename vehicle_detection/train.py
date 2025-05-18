import sys, ast, pathlib, os, torch, pandas as pd, torchvision
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.append(str(pathlib.Path(__file__).parent / "src"))
import utils
from engine import train_one_epoch, evaluate

EPOCH = 10


def load_label_map(txt_path="data/category.txt"):
    raw = pathlib.Path(txt_path).read_text().strip()

    # 如果前面有 "category:"，就把冒號後面的部分切出來
    if raw.lower().startswith("category"):
        raw = raw.split(":", 1)[1].strip()

    d = ast.literal_eval(raw)  # 變成 {'Bus':0, ...}
    return {k: v + 1 for k, v in d.items()}  # 背景留 0，車類 +1 → 1~5


LABEL_MAP = load_label_map()  # 全域一次即可
NUM_CLASSES = max(LABEL_MAP.values()) + 1  # 背景 0 + 5 類 = 6


def get_model(num_classes: int = NUM_CLASSES):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


class VehicleDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        df = pd.read_csv(csv_file)
        df["label_id"] = df["class"].map(LABEL_MAP).astype(int)
        self.groups = df.groupby("filename")
        self.fnames = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        rows = self.groups.get_group(fname)
        img = tv_tensors.Image(read_image(os.path.join(self.img_dir, fname)))

        boxes = torch.as_tensor(
            rows[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
        )
        labels = torch.as_tensor(rows["label_id"].values, dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=F.get_size(img)
            ),
            "labels": labels,
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def main():
    collate_fn = utils.collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = VehicleDataset(
        "data/train_labels.csv", "data/train", get_transform(train=True)
    )
    test_ds = VehicleDataset(
        "data/test_labels.csv", "data/test", get_transform(train=False)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    model = get_model(num_classes=6).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=5e-4)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(EPOCH):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_sched.step()
        evaluate(model, test_loader, device=device)

    torch.save(model.state_dict(), "outputs/fasterrcnn.pth")

    # visualize
    # ---------- 1. 反轉 label 對照表（id -> 中文名） ----------
    ID2NAME = {v: k for k, v in LABEL_MAP.items()}  # {1:'Bus', 2:'Car', ...}

    # ---------- 2. 載入模型權重，切 eval 模式 ----------
    model.eval()
    # 若你剛剛已經 torch.save(model.state_dict(), ...)，
    #   也可以重新載入:  model.load_state_dict(torch.load('outputs/fasterrcnn.pth'))

    # ---------- 3. 隨便選一張測試圖片 ----------
    img_path = "data/test/frame_3403_jpg.rf.30d0e1ebeda1a2d34985cee7e69fedeb.jpg"  # ← 換成你的實際檔名
    img = read_image(img_path)  # uint8, [C,H,W]
    H, W = img.shape[1:]

    with torch.no_grad():
        pred = model([img.to(device).float() / 255.0])[0]  # → dict

    # ---------- 4. 過濾高分框 ----------
    keep = pred["scores"] >= 0.5
    boxes = pred["boxes"][keep].cpu().round().int()
    labels = pred["labels"][keep].cpu()
    scores = pred["scores"][keep].cpu()

    # ---------- 5. 組合顯示文字 ----------
    text = [f"{ID2NAME[int(lbl)]}: {s:.2f}" for lbl, s in zip(labels, scores)]

    # torchvision.utils 只吃 uint8 Tensor，且要在 CPU
    from torchvision.utils import draw_bounding_boxes

    img_drawn = draw_bounding_boxes(img, boxes, labels=text, colors="red", width=2)

    # ---------- 6. 顯示 & 存檔 ----------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(img_drawn.permute(1, 2, 0))  # C,H,W → H,W,C
    plt.axis("off")
    plt.title("模型偵測結果 (score ≥ 0.5)")
    plt.show()

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "prediction_vis.png")
    torchvision.io.write_png(img_drawn, out_path)
    print(f"✅ 已儲存標註圖：{out_path}")


if __name__ == "__main__":
    main()
