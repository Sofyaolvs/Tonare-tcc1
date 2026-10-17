"""
Teste EXTERNO no Fitzpatrick17k (foto clínica de atlas online): melanoma vs nevo melanocítico,
por grupo de tom de pele (escala Fitzpatrick do próprio dataset). Nada daqui entra no treino.

Definição (decisão de análise, não do dataset):
  melanoma  = rótulos "melanoma", "superficial spreading melanoma ssm", "malignant melanoma"
  nevo      = "nevocytic nevus", "halo nevus", "congenital nevus"
  lentigo maligna (83 imagens, TODAS do Atlas Dermatológico, outra fonte) fica à parte: se entrasse
  junto, a fonte da imagem viraria atalho.
Limitações: rótulos de atlas, sem biópsia garantida; sem id de lesão, então o bootstrap é por imagem
(IC otimista se houver várias fotos da mesma lesão); duas anotações independentes de tom
(fitzpatrick_scale x fitzpatrick_centaur) só concordam em ~46% das imagens, então o tom é ruidoso.

Pré-requisitos: data/raw/fitzpatrick17k/fitzpatrick17k.csv (repo dos autores) e o zip das imagens.

Uso:
    python scripts/evaluate_fitzpatrick17k.py v6 [caminho/fitzpatrick17k.zip]
"""

from __future__ import annotations

import io
import sys
import zipfile
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
RAW = BASE_DIR / "data" / "raw" / "fitzpatrick17k"
OUT_DIR = BASE_DIR / "data" / "processed" / "fitzpatrick17k"
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v6"
ZIP_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Downloads" / "fitzpatrick17k.zip"
CHECKPOINT = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{VERSION}.pt"

MEL = ["melanoma", "superficial spreading melanoma ssm", "malignant melanoma"]
NEV = ["nevocytic nevus", "halo nevus", "congenital nevus"]
LM = ["lentigo maligna"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_BOOT, SEED, MIN_N = 2000, 42, 10


def grupo(f: float) -> str:
    return "sem tom" if f < 0 else ("I-II" if f <= 2 else ("III-IV" if f <= 4 else "V-VI"))


def ensure_images(ids: list[str]) -> Path:
    out = RAW / "images"
    out.mkdir(parents=True, exist_ok=True)
    faltam = [i for i in ids if not (out / f"{i}.jpg").exists()]
    if faltam:
        with zipfile.ZipFile(ZIP_PATH) as z:
            by_name = {Path(n).name: n for n in z.namelist() if n.endswith(".jpg")}
            for i in faltam:
                (out / f"{i}.jpg").write_bytes(z.read(by_name[f"{i}.jpg"]))  # nome = md5 de 32 hex, sem caminho
    return out


def preprocess_like_training(img: Image.Image) -> Image.Image:
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def sens_spec(y, s, t):
    pred = s >= t
    return (float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
            float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan"))


def boot(y, s, t, rng):
    a, se, sp = [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], s[i]))
        e, p = sens_spec(y[i], s[i], t)
        se.append(e)
        sp.append(p)
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return q(a), q(se), q(sp)


def main() -> None:
    d = pd.read_csv(RAW / "fitzpatrick17k.csv")
    d = d[d["label"].isin(MEL + NEV + LM)].copy()
    d["grupo"] = d["fitzpatrick_scale"].map(grupo)
    d["y"] = d["label"].isin(MEL + LM).astype(int)
    man7 = BASE_DIR / "data" / "processed" / "split_index_v7.csv"
    if VERSION == "v7" and man7.exists():
        usadas = set(pd.read_csv(man7).query("dataset == 'fitzpatrick17k'")["source_image_id"])
        d = d[~d["md5hash"].isin(usadas)].copy()  # v7 treinou/validou com os tons I-IV: só o resto é teste
        print(f"v7: {len(usadas)} imagens do Fitzpatrick17k entraram no treino/validação e foram EXCLUÍDAS do teste.")
    img_dir = ensure_images(list(d["md5hash"]))

    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    thr = float(ck["threshold"])

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    paths = [img_dir / f"{i}.jpg" for i in d["md5hash"]]
    s = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            batch = torch.stack([tf(preprocess_like_training(Image.open(p))) for p in paths[i:i + 32]])
            s.extend(torch.softmax(model(batch), dim=1)[:, 1].tolist())
    d["score_mel"] = s
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d[["md5hash", "label", "fitzpatrick_scale", "grupo", "y", "score_mel"]].to_csv(OUT_DIR / f"fitz17k_{VERSION}.csv", index=False)

    rng = np.random.default_rng(SEED)
    core = d[d["label"].isin(MEL + NEV)]
    f = lambda v, ci: f"{v:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"
    print(f"Fitzpatrick17k | {VERSION}, limiar {thr:.2f} | melanoma ({len(MEL)} rótulos) vs nevo melanocítico ({len(NEV)} rótulos)")
    print(f"{'tom':<10}{'mel':>5}{'nv':>5}  {'AUC [IC95%]':<24}{'sensib. [IC95%]':<24}{'especif. [IC95%]':<24}")
    for nome, g in [("todos", core)] + [(k, core[core["grupo"] == k]) for k in ("I-II", "III-IV", "V-VI")]:
        y, sc = g["y"].to_numpy(), g["score_mel"].to_numpy()
        n_mel, n_nv = int(y.sum()), int((y == 0).sum())
        if min(n_mel, n_nv) < 2:
            continue
        se, sp = sens_spec(y, sc, thr)
        a_ci, se_ci, sp_ci = boot(y, sc, thr, rng)
        flag = "" if min(n_mel, n_nv) >= MIN_N else " *"
        print(f"{nome:<10}{n_mel:>5}{n_nv:>5}  {f(roc_auc_score(y, sc), a_ci):<24}{f(se, se_ci):<24}{f(sp, sp_ci):<24}{flag}")
    lm = d[d["label"].isin(LM)]
    print(f"\nLentigo maligna à parte ({len(lm)} imagens, todas do Atlas Dermatológico): "
          f"detectadas no limiar = {int((lm['score_mel'] >= thr).sum())} de {len(lm)} "
          f"({(lm['score_mel'] >= thr).mean():.2f})")
    print("\nSensibilidade só nos melanomas por tom (descritivo): ")
    mel = core[core["y"] == 1]
    print(mel.groupby("grupo")["score_mel"].agg(n="count", mediana="median", detectados=lambda x: int((x >= thr).sum())).round(3).to_string())
    print(f"\n('*' = menos de {MIN_N} casos em alguma classe, só descritivo.)  Previsões: {OUT_DIR / f'fitz17k_{VERSION}.csv'}")


if __name__ == "__main__":
    main()
