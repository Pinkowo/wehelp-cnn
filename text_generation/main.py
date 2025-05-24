#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build & train a character-level Transformer LM on ancient-Chinese corpus, then
generate at least 5 sentences.

Run:
    python train_transformer.py --epochs 30 --cuda         # 若有 GPU
    python train_transformer.py                            # CPU 版
"""

import json, math,  argparse
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import textwrap

# ---------- Hyper-parameters ----------
EMBED  = 264     # embedding dim
NLAYER = 6        # transformer layers
NHEAD  = 8        # attention heads
FFN    = 1024      # feed-forward dim
SEQ_LN = 120      # training block length
BATCH  = 32
LR     = 1e-4
EPOCHS = 30
GENERATE_LEN = 240

# ---------- Data loading -------------
corp_path = Path(__file__).with_name("data.json")
text = []
with open(corp_path, "r", encoding="utf-8") as f:
    data = json.load(f)
for chapter in data:
    for para in chapter["paragraphs"]:
        text.append(para.strip())
full_text = "。".join(text)  # 粗略串接，保留古文節奏
chars = sorted(set(full_text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)

def encode(s): return [stoi[c] for c in s]
def decode(t): return "".join(itos[i] for i in t)

# ---------- Dataset ------------------
class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        ids = torch.tensor(encode(data), dtype=torch.long)
        self.x = ids[:-1]
        self.y = ids[1:]
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.x) - 1) // self.seq_len

    def __getitem__(self, idx):
        i = idx * self.seq_len
        return (self.x[i: i + self.seq_len],
                self.y[i: i + self.seq_len])

dataset = CharDataset(full_text, SEQ_LN)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# ---------- Model ---------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)      # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED)
        self.pos   = PositionalEncoding(EMBED)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED, nhead=NHEAD, dim_feedforward=FFN, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, NLAYER)
        self.head = nn.Linear(EMBED, vocab_size)

    def forward(self, x):
        # x: (B, T)
        x = self.embed(x)
        x = self.pos(x)

        T = x.size(1)
        # ① 產生 causal mask，True 代表要遮
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        # ② 傳進 transformer
        x = self.transformer(x, mask=causal_mask)   # ← 把 src_mask 換成 mask
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt, max_new=240, T=0.9, top_k=40, penalty=1.2):
        self.eval()
        ids = encode(prompt)
        freq = {}                          # 記錄已出現次數
        device = next(self.parameters()).device
        for _ in range(max_new):
            ctx = torch.tensor([ids[-SEQ_LN:]], dtype=torch.long, device=device)
            logits = self(ctx)[0, -1] / T   # (vocab,)
            # 重複懲罰
            for tok, cnt in freq.items():
                logits[tok] -= penalty * cnt
            # top-k
            topk_val, topk_idx = logits.topk(top_k)
            probs = torch.softmax(topk_val, dim=-1)
            idx = topk_idx[torch.multinomial(probs, 1)]
            idx = idx.item()
            ids.append(idx)
            freq[idx] = freq.get(idx, 0) + 1
        return decode(ids)


# ---------- Train loop ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model  = TransformerLM().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}/{args.epochs}  –  loss: {total_loss/len(loader):.4f}")

    # ---------- Generate & save -------------
    prompt = "天下"
    gen = model.generate(prompt)
    print("\n=== Generated text ===\n")
    print(textwrap.fill(gen, width=60))

    # 擷取前 5 句（以「。」「？」或「！」斷句）
    sentences = [s for s in gen.split("。") if len(s) > 5][:5]
    print("\n--- 5 sentences ---")
    for i, s in enumerate(sentences, 1):
        print(f"{i}. {s}。")

if __name__ == "__main__":
    main()
