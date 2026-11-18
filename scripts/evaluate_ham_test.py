"""
Teste do HAM10000 (dermatoscopia, o mesmo teste de todas as versões) com IC95% por bootstrap de LESÃO.
Serve para comparar v2 x v4 x v5 com o mesmo pipeline.

Uso:
    python scripts/evaluate_ham_test.py v2
    python scripts/evaluate_ham_test.py v5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
CHECKPOINT = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{VERSION}.pt"
MANIFEST = BASE_DIR / "data" / "processed" / "split_index.csv"  # teste do HAM10000, idêntico em v2/v4/v5
N_BOOT, SEED = 2000, 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def main() -> None:
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    thr = float(ck["threshold"])

    df = pd.read_csv(MANIFEST)
    df = df[(df["split"] == "test") & (~df["is_augmented"])].reset_index(drop=True)
    df["y"] = (df["dx"] == "mel").astype(int)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    paths = [BASE_DIR / p.replace("\\", "/") for p in df["path"]]
    probs = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]])
            probs.extend(torch.softmax(model(batch), dim=1)[:, 1].tolist())
    df["score"] = probs
    y, s = df["y"].to_numpy(), df["score"].to_numpy()

    groups = list(df.groupby("lesion_id").indices.values())
    rng = np.random.default_rng(SEED)
    a, se, sp = [], [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yb, sb = y[idx], s[idx]
        if yb.min() == yb.max():
            continue
        a.append(roc_auc_score(yb, sb))
        pred = sb >= thr
        se.append(pred[yb == 1].mean())
        sp.append((~pred[yb == 0]).mean())
    q = lambda v: f"[{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}]"
    pred = s >= thr
    print(f"HAM10000 teste ({len(df)} imgs, {int(y.sum())} mel) | {VERSION}, limiar {thr:.2f}")
    print(f"  AUC            {roc_auc_score(y, s):.3f} {q(a)}")
    print(f"  sensibilidade  {pred[y == 1].mean():.3f} {q(se)}")
    print(f"  especificidade {(~pred[y == 0]).mean():.3f} {q(sp)}")


if __name__ == "__main__":
    main()
