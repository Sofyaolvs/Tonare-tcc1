"""
Sprint 3 (extensão) — Avaliação no MILK10k: modelo treinado no HAM10000 (mel vs nv) aplicado a
lesões mel/nv do MILK10k, separado por tom de pele E por tipo de imagem.

O MILK10k traz, para cada lesão, a foto clínica (close-up) e a dermatoscopia. Isso permite separar
dois efeitos que no DDI estavam misturados:
  - dermatoscopia (mesmo tipo de imagem do treino) -> mede tom de pele "limpo"
  - foto clínica (tipo diferente do treino)       -> mede o efeito da troca de modalidade
Só entram lesões mel e nv (a tarefa do modelo); rótulo vem do ground truth do ISIC (biópsia na
maioria). Escala de tom de pele do MILK10k: 0 = muito escuro ... 5 = muito claro (NÃO é Fitzpatrick).
Bootstrap por LESÃO (a foto clínica e a dermatoscopia da mesma lesão não são independentes).

Uso:
    python scripts/evaluate_milk10k.py                # v2, todas as lesões mel/nv (nenhuma entrou no treino da v2)
    python scripts/evaluate_milk10k.py v2 heldout     # v2, só as lesões "test_milk" do manifesto v4
    python scripts/evaluate_milk10k.py v4 heldout     # v4, mesmas lesões (comparação justa v2 x v4)
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
RAW_DIR = BASE_DIR / "data" / "raw" / "milk10k"
OUT_DIR = BASE_DIR / "data" / "processed" / "milk10k"
CHECKPOINT_VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
HELDOUT = len(sys.argv) > 2 and sys.argv[2] == "heldout"
TAG = f"_{CHECKPOINT_VERSION}" + ("_heldout" if HELDOUT else "")
CHECKPOINT = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{CHECKPOINT_VERSION}.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_BOOT = 2000
SEED = 42
MIN_N_POR_CLASSE = 10  # abaixo disso, a métrica do grupo é só descritiva
TONE_GROUPS = {"escuro (0-2)": [0, 1, 2], "médio (3)": [3], "claro (4-5)": [4, 5]}


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, float(ck["threshold"])


def preprocess_like_training(img: Image.Image) -> Image.Image:
    # Mesmo pipeline do scripts/preprocess.py: resize bilinear 224x224 + JPEG q95.
    # A recompressão importa: sem ela o modelo vê pixels sutilmente diferentes do treino.
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@torch.no_grad()
def score_zip(model, zf: zipfile.ZipFile, members: list[str], batch_size: int = 32) -> np.ndarray:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    out = []
    for i in range(0, len(members), batch_size):
        imgs = []
        for m in members[i:i + batch_size]:
            with zf.open(m) as f:
                imgs.append(tf(preprocess_like_training(Image.open(f))))
        out.extend(torch.softmax(model(torch.stack(imgs)), dim=1)[:, 1].tolist())
        print(f"  {min(i + batch_size, len(members))}/{len(members)}", end="\r")
    print()
    return np.array(out)


def sens_spec(y, s, thr):
    pred = s >= thr
    pos, neg = y == 1, y == 0
    return (float(pred[pos].mean()) if pos.any() else float("nan"),
            float((~pred[neg]).mean()) if neg.any() else float("nan"))


def lesion_bootstrap(g: pd.DataFrame, thr: float, rng) -> dict:
    y_all, s_all = g["y"].to_numpy(), g["score_mel"].to_numpy()
    idx_by_lesion = list(g.reset_index(drop=True).groupby("lesion_id").indices.values())
    aucs, sens_l, spec_l = [], [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(idx_by_lesion), len(idx_by_lesion))
        idx = np.concatenate([idx_by_lesion[i] for i in pick])
        y, s = y_all[idx], s_all[idx]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, s))
        se, sp = sens_spec(y, s, thr)
        sens_l.append(se)
        spec_l.append(sp)
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return {"auc": q(aucs), "sens": q(sens_l), "spec": q(spec_l)}


def main() -> None:
    gt = pd.read_csv(RAW_DIR / "training_gt.csv")
    inp = pd.read_csv(RAW_DIR / "training_input.csv")
    gt["dx"] = gt[[c for c in gt.columns if c != "lesion_id"]].idxmax(axis=1)
    df = inp.merge(gt[["lesion_id", "dx"]], on="lesion_id")
    df = df[df["dx"].isin(["MEL", "NV"])].copy()
    df["y"] = (df["dx"] == "MEL").astype(int)
    df["modalidade"] = np.where(df["image_type"].str.contains("dermoscopic"), "dermatoscopia", "foto clínica")

    zf = zipfile.ZipFile(RAW_DIR / "milk10k.zip")
    by_id = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    df = df[df["isic_id"].isin(by_id)].copy()
    if HELDOUT:
        man = pd.read_csv(BASE_DIR / "data" / "processed" / "split_index_v4.csv")
        df = df[df["lesion_id"].isin(man[man["split"] == "test_milk"]["lesion_id"])].copy()
    print(f"{len(df)} imagens ({df['lesion_id'].nunique()} lesões mel/nv) encontradas no zip.")

    model, thr = load_model(CHECKPOINT)
    print(f"Checkpoint {CHECKPOINT_VERSION}, limiar {thr:.2f} (escolhido na validação do HAM10000)")
    df["score_mel"] = score_zip(model, zf, [by_id[i] for i in df["isic_id"]])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"milk10k_predictions{TAG}.csv", index=False)

    rng = np.random.default_rng(SEED)
    rows = []
    for mod, gm in [("TODAS", df)] + list(df.groupby("modalidade")):
        grupos = [("todos os tons", gm)] + [(n, gm[gm["skin_tone_class"].isin(t)]) for n, t in TONE_GROUPS.items()]
        for nome, g in grupos:
            n_mel, n_nv = int(g["y"].sum()), int((g["y"] == 0).sum())
            if n_mel == 0 or n_nv == 0:
                continue
            y, s = g["y"].to_numpy(), g["score_mel"].to_numpy()
            se, sp = sens_spec(y, s, thr)
            ci = lesion_bootstrap(g, thr, rng)
            rows.append({"modalidade": mod, "tom": nome, "n_mel": n_mel, "n_nv": n_nv,
                         "auc": roc_auc_score(y, s), "auc_ic": ci["auc"],
                         "sens": se, "sens_ic": ci["sens"], "spec": sp, "spec_ic": ci["spec"],
                         "robusto": min(n_mel, n_nv) >= MIN_N_POR_CLASSE})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / f"milk10k_fairness_summary{TAG}.csv", index=False)

    f = lambda v, ci: f"{v:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"
    print(f"\nMILK10k mel vs nv, limiar {thr:.2f}. IC95% por bootstrap de lesões. '*' = grupo pequeno (<{MIN_N_POR_CLASSE} em alguma classe), só descritivo.")
    print(f"{'modalidade':<15}{'tom':<16}{'mel':>5}{'nv':>5}  {'AUC [IC95%]':<24}{'sensib. [IC95%]':<24}{'especif. [IC95%]':<24}")
    for r in rows:
        flag = "" if r["robusto"] else " *"
        print(f"{r['modalidade']:<15}{r['tom']:<16}{r['n_mel']:>5}{r['n_nv']:>5}  {f(r['auc'], r['auc_ic']):<24}"
              f"{f(r['sens'], r['sens_ic']):<24}{f(r['spec'], r['spec_ic']):<24}{flag}")

    print("\nDiferença entre tons, por modalidade (bootstrap de lesões independente por grupo; IC95% que exclui 0 = sustentada):")
    prep = {}
    for mod, gm in df.groupby("modalidade"):
        for nome, t in TONE_GROUPS.items():
            g = gm[gm["skin_tone_class"].isin(t)].reset_index(drop=True)
            prep[(mod, nome)] = (g["y"].to_numpy(), g["score_mel"].to_numpy(),
                                 list(g.groupby("lesion_id").indices.values()))

    def draw(key):
        y, s, idx = prep[key]
        pick = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx[i] for i in pick])
        return y[ii], s[ii]

    for mod in df["modalidade"].unique():
        for a, b in [("escuro (0-2)", "claro (4-5)"), ("escuro (0-2)", "médio (3)")]:
            d_auc, d_sens = [], []
            for _ in range(N_BOOT):
                ya, sa = draw((mod, a))
                yb, sb = draw((mod, b))
                if ya.min() == ya.max() or yb.min() == yb.max():
                    continue
                d_auc.append(roc_auc_score(ya, sa) - roc_auc_score(yb, sb))
                d_sens.append(sens_spec(ya, sa, thr)[0] - sens_spec(yb, sb, thr)[0])
            for nome_m, d in [("AUC", d_auc), ("sensibilidade", d_sens)]:
                d = np.array(d)
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                print(f"  {mod:<14} {a} - {b}  {nome_m:<14}{d.mean():+.3f} [{np.percentile(d, 2.5):+.3f} a {np.percentile(d, 97.5):+.3f}] p~{p:.3f}")

    print(f"\nResumo: {OUT_DIR / f'milk10k_fairness_summary{TAG}.csv'}\nPrevisões: {OUT_DIR / f'milk10k_predictions{TAG}.csv'}")


if __name__ == "__main__":
    main()
