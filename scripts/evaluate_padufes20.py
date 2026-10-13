"""
Teste EXTERNO em fotos de celular: modelo (HAM10000 / MILK10k) aplicado ao PAD-UFES-20
(Brasil, UFES), lesões mel vs nv. Nada do PAD-UFES-20 entra no treino: é uma fonte independente
(outro país, outro tipo de câmera) para responder "isso generaliza para foto clínica de verdade?".

Limitações (ver README): só 52 imagens / 36 lesões de melanoma, nenhum em Fitzpatrick V-VI;
só ~25% dos nevos têm biópsia (o resto é diagnóstico clínico). Bootstrap por lesão.

Pré-requisito: data/raw/padufes20/{metadata.csv,images/} (só mel e nv extraídos).

Uso:
    python scripts/evaluate_padufes20.py        # v2
    python scripts/evaluate_padufes20.py v4
"""

from __future__ import annotations

import io
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
RAW_DIR = BASE_DIR / "data" / "raw" / "padufes20"
OUT_DIR = BASE_DIR / "data" / "processed" / "padufes20"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
CHECKPOINT = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{VERSION}.pt"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_BOOT, SEED = 2000, 42


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, float(ck["threshold"])


def preprocess_like_training(img: Image.Image) -> Image.Image:
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def score(model, paths: list[Path]) -> np.ndarray:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    out = []
    for i in range(0, len(paths), 32):
        batch = torch.stack([tf(preprocess_like_training(Image.open(p))) for p in paths[i:i + 32]])
        out.extend(torch.softmax(model(batch), dim=1)[:, 1].tolist())
    return np.array(out)


def sens_spec(y, s, thr):
    pred = s >= thr
    return (float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
            float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan"))


def boot(df: pd.DataFrame, thr: float, rng) -> dict:
    y_all, s_all = df["y"].to_numpy(), df["score_mel"].to_numpy()
    groups = list(df.reset_index(drop=True).groupby("lesion_id").indices.values())
    a, se, sp = [], [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y, s = y_all[idx], s_all[idx]
        if y.min() == y.max():
            continue
        a.append(roc_auc_score(y, s))
        e, p = sens_spec(y, s, thr)
        se.append(e)
        sp.append(p)
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return {"auc": q(a), "sens": q(se), "spec": q(sp)}


def main() -> None:
    meta = pd.read_csv(RAW_DIR / "metadata.csv")
    df = meta[meta["diagnostic"].isin(["MEL", "NEV"])].copy()
    df["y"] = (df["diagnostic"] == "MEL").astype(int)
    df["path"] = [RAW_DIR / "images" / i for i in df["img_id"]]

    model, thr = load_model(CHECKPOINT)
    print(f"Checkpoint {VERSION}, limiar {thr:.2f} | {len(df)} imagens, {df['lesion_id'].nunique()} lesões "
          f"({int(df['y'].sum())} imagens de melanoma)")
    df["score_mel"] = score(model, list(df["path"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns="path").to_csv(OUT_DIR / f"padufes20_predictions_{VERSION}.csv", index=False)

    rng = np.random.default_rng(SEED)
    f = lambda v, ci: f"{v:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"
    y, s = df["y"].to_numpy(), df["score_mel"].to_numpy()
    se, sp = sens_spec(y, s, thr)
    ci = boot(df, thr, rng)
    print(f"\nPAD-UFES-20 (foto de celular), mel vs nv, limiar {thr:.2f}; IC95% por bootstrap de lesões")
    print(f"  AUC            {f(roc_auc_score(y, s), ci['auc'])}")
    print(f"  sensibilidade  {f(se, ci['sens'])}   (melanomas detectados: {int(((s >= thr) & (y == 1)).sum())} de {int(y.sum())} imagens)")
    print(f"  especificidade {f(sp, ci['spec'])}")

    print("\nMelanomas por Fitzpatrick (descritivo, n pequeno demais p/ métrica):")
    mel = df[df["y"] == 1]
    print(mel.groupby(mel["fitspatrick"].fillna(-1).astype(int))["score_mel"].agg(
        n="count", mediana="median", detectados=lambda x: int((x >= thr).sum())).round(3))


if __name__ == "__main__":
    main()
