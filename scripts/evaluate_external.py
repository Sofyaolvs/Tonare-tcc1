"""
Testes externos em coleções do ISIC (nunca usadas no treino): mel vs nv.

  hiba251             Hospital Italiano de Buenos Aires — dermatoscopia + foto clínica (Fitzpatrick I-IV)
  consec_mel2020_216  MSK, biópsias consecutivas de 2020 — dermatoscopia, população de encaminhamento real

Pré-requisito: scripts/download_isic_images.py <nome> (imagens em data/raw/candidates/images/<nome>/).
IC95% por bootstrap de LESÃO quando há lesion_id; senão por imagem (cada imagem = uma unidade).

Uso:
    python scripts/evaluate_external.py v2 hiba251
    python scripts/evaluate_external.py v4 consec_mel2020_216
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
CAND = BASE_DIR / "data" / "raw" / "candidates"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
NAME = sys.argv[2] if len(sys.argv) > 2 else "hiba251"
CHECKPOINT = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{VERSION}.pt"
OUT_DIR = BASE_DIR / "data" / "processed" / "external"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_BOOT, SEED, MIN_N = 2000, 42, 10


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


def boot(g: pd.DataFrame, thr: float, rng) -> dict:
    y_all, s_all = g["y"].to_numpy(), g["score_mel"].to_numpy()
    groups = list(g.reset_index(drop=True).groupby("grp").indices.values())
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
    d = pd.read_csv(CAND / f"{NAME}.csv", low_memory=False)
    d["y"] = (~d["diagnosis_3"].astype(str).str.contains("Nevus")).astype(int)
    d["modalidade"] = np.where(d["acq_image_type"].astype(str).str.contains("clinical"), "foto clínica", "dermatoscopia")
    d["grp"] = d["lesion_id"] if "lesion_id" in d and d["lesion_id"].notna().all() else d["isic_id"]
    img_dir = CAND / "images" / NAME
    d = d[d["isic_id"].map(lambda i: (img_dir / f"{i}.jpg").exists())].copy()

    model, thr = load_model(CHECKPOINT)
    print(f"{NAME} | checkpoint {VERSION}, limiar {thr:.2f} | {len(d)} imagens, {d['grp'].nunique()} unidades "
          f"({int(d['y'].sum())} mel)")
    d["score_mel"] = score(model, [img_dir / f"{i}.jpg" for i in d["isic_id"]])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d[["isic_id", "grp", "y", "modalidade", "score_mel"]].to_csv(OUT_DIR / f"{NAME}_{VERSION}.csv", index=False)

    rng = np.random.default_rng(SEED)
    f = lambda v, ci: f"{v:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"
    print(f"\n{'modalidade':<15}{'mel':>5}{'nv':>5}  {'AUC [IC95%]':<24}{'sensib. [IC95%]':<24}{'especif. [IC95%]':<24}")
    for mod, g in [("TODAS", d)] + list(d.groupby("modalidade")):
        n_mel, n_nv = int(g["y"].sum()), int((g["y"] == 0).sum())
        if n_mel < 2 or n_nv < 2:
            continue
        y, s = g["y"].to_numpy(), g["score_mel"].to_numpy()
        se, sp = sens_spec(y, s, thr)
        ci = boot(g, thr, rng)
        flag = "" if min(n_mel, n_nv) >= MIN_N else " *"
        print(f"{mod:<15}{n_mel:>5}{n_nv:>5}  {f(roc_auc_score(y, s), ci['auc']):<24}{f(se, ci['sens']):<24}{f(sp, ci['spec']):<24}{flag}")
    print(f"\n('*' = poucos casos em alguma classe, só descritivo.) Previsões: {OUT_DIR / f'{NAME}_{VERSION}.csv'}")


if __name__ == "__main__":
    main()
