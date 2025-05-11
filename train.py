import argparse, random, os, numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------- utils ---------- #
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------- dataset ---------- #
class HandwritingDS(Dataset):
    def __init__(self, csv, root_dir, img_root, img_size=32):
        df = pd.read_csv(csv)

        files = df["filename"].tolist()
        labels = df["label"].tolist()

        keep = []
        base = Path(root_dir) / img_root
        for f, l in zip(files, labels):
            if (base / f).is_file():
                keep.append((f, l))

        self.items = keep
        self.c2i = {c: i for i, c in enumerate(sorted({l for _, l in keep}))}
        self.root = base
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f, l = self.items[idx]
        img = Image.open(self.root / f).convert("L")
        return self.tf(img), self.c2i[l]


class FolderDataset(Dataset):
    def __init__(self, root_dir, img_size=32):
        self.root = Path(root_dir)
        self.files = list(self.root.rglob("*.png"))
        labels = sorted({p.parent.name for p in self.files})
        self.c2i = {c: i for i, c in enumerate(labels)}
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(f).convert("L")
        label = self.c2i[f.parent.name]
        return self.tf(img), label


# ---------- model ---------- #
class SimpleCNN(nn.Module):
    def __init__(self, n_cls=62):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.net(x)


# ---------- train / eval ---------- #
def run(loader, model, crit, opt=None, device="cpu"):
    if opt:
        model.train()
    else:
        model.eval()
    tot, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if opt:
            opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        if opt:
            loss.backward()
            opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return loss_sum / tot, correct / tot


def main(cfg):
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg.data_dir)
    csv = root / "handwriting/image_labels.csv"

    train_ds = HandwritingDS(
        csv=root / "handwriting/image_labels.csv",
        root_dir=root,
        img_root="handwriting/augmented_images",
    )

    test_ds = FolderDataset(root / "handwriting/combined_folder/test")
    print("train samples :", len(train_ds))
    print("test  samples :", len(test_ds))

    tr_dl = DataLoader(train_ds, cfg.bs, shuffle=True, num_workers=2)
    te_dl = DataLoader(test_ds, cfg.bs, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best = 0
    for ep in range(1, cfg.epochs + 1):
        _, tr_acc = run(tr_dl, model, crit, opt, device)
        _, te_acc = run(te_dl, model, crit, None, device)
        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), "best.pth")
        print(f"[{ep:02d}] train {tr_acc:.2%} | test {te_acc:.2%} (best {best:.2%})")
    print("Done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    cfg = ap.parse_args()
    main(cfg)
