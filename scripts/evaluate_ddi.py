"""
Sprint 3 — Avaliação de fairness: modelo treinado no HAM10000 (mel vs nv) aplicado ao DDI,
separado por faixa de tom de pele.

O modelo dá P(melanoma) para cada imagem. No DDI o rótulo é "maligno vs benigno" (78 doenças),
então P(mel) é usada como escore de malignidade. ATENÇÃO à limitação metodológica: o HAM10000
é dermatoscópico e o DDI é fotografia clínica — parte da queda de desempenho vem dessa troca de
tipo de imagem, não do tom de pele. Por isso a comparação válida é ENTRE as faixas de tom de
pele (todas sofrem a mesma troca), não o valor absoluto.

Métricas por faixa (com IC 95% por bootstrap dentro de cada faixa, pois os grupos são pequenos):
  - AUC-ROC (não depende de limiar)
  - sensibilidade e especificidade no limiar salvo no checkpoint (escolhido na validação do HAM10000)

Uso:
    python scripts/evaluate_ddi.py
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
# v2 = referência (recomendada, DDI interpretável); v3 = achado sobre generalização cross-modalidade
# (ver memória do projeto). Nomes de arquivo explícitos de propósito - um checkpoint sem sufixo de
# versão já causou um acidente aqui (uma corrida sobrescreveu/apagou a outra sem querer).
CHECKPOINT_VERSION = "v2"
CHECKPOINT = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{CHECKPOINT_VERSION}.pt"
DDI_MANIFEST = BASE_DIR / "data" / "processed" / "ddi" / "ddi_manifest.csv"
DDI_RAW_META = BASE_DIR / "data" / "raw" / "ddi" / "ddi_metadata.csv"
PRED_PATH = BASE_DIR / "data" / "processed" / "ddi" / "ddi_predictions.csv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "ddi" / "ddi_fairness_summary.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_BOOT = 2000
SEED = 42


def load_model(checkpoint_path: Path):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, float(ck["threshold"])


@torch.no_grad()
def score_images(model, paths: list[Path], batch_size: int = 32) -> np.ndarray:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    probs = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
        probs.extend(torch.softmax(model(batch), dim=1)[:, 1].tolist())
    return np.array(probs)


def sens_spec(y: np.ndarray, score: np.ndarray, thr: float) -> tuple[float, float]:
    pred = score >= thr
    pos, neg = y == 1, y == 0
    sens = pred[pos].mean() if pos.any() else float("nan")
    spec = (~pred[neg]).mean() if neg.any() else float("nan")
    return float(sens), float(spec)


def bootstrap_ci(y: np.ndarray, score: np.ndarray, thr: float, rng: np.random.Generator) -> dict:
    aucs, sens_l, spec_l = [], [], []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], score[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, sb))
        s, sp = sens_spec(yb, sb, thr)
        sens_l.append(s)
        spec_l.append(sp)
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return {"auc_ci": q(aucs), "sens_ci": q(sens_l), "spec_ci": q(spec_l)}


def main() -> None:
    model, thr = load_model(CHECKPOINT)
    print(f"Checkpoint carregado. Limiar (escolhido na validação do HAM10000): {thr:.2f}")

    df = pd.read_csv(DDI_MANIFEST)
    raw = pd.read_csv(DDI_RAW_META)[["DDI_file", "disease"]].rename(columns={"DDI_file": "ddi_file"})
    df = df.merge(raw, on="ddi_file", how="left")
    df["y"] = df["malignant"].astype(int)

    paths = [BASE_DIR / p.replace("\\", "/") for p in df["path"]]
    print(f"Pontuando {len(paths)} imagens do DDI (CPU)...")
    df["score_mel"] = score_images(model, paths)
    df.to_csv(PRED_PATH, index=False)

    rng = np.random.default_rng(SEED)
    rows = []
    grupos = [("TODOS", df)] + [(b, g) for b, g in df.groupby("skin_tone_band")]
    for nome, g in grupos:
        y, s = g["y"].to_numpy(), g["score_mel"].to_numpy()
        sens, spec = sens_spec(y, s, thr)
        ci = bootstrap_ci(y, s, thr, rng)
        rows.append({
            "grupo": nome, "n": len(g), "n_malignos": int(y.sum()),
            "auc": roc_auc_score(y, s), "auc_ic95": ci["auc_ci"],
            "sensibilidade": sens, "sens_ic95": ci["sens_ci"],
            "especificidade": spec, "espec_ic95": ci["spec_ci"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    fmt = lambda v, ci: f"{v:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"
    print(f"\nDDI (maligno vs benigno) com o modelo mel-vs-nv do HAM10000, limiar {thr:.2f}")
    print(f"{'grupo':<22}{'n':>5}{'malig.':>7}  {'AUC [IC95%]':<24}{'sensibilidade [IC95%]':<26}{'especificidade [IC95%]':<26}")
    for r in rows:
        print(f"{r['grupo']:<22}{r['n']:>5}{r['n_malignos']:>7}  {fmt(r['auc'], r['auc_ic95']):<24}"
              f"{fmt(r['sensibilidade'], r['sens_ic95']):<26}{fmt(r['especificidade'], r['espec_ic95']):<26}")

    print("\nDiferença de AUC entre faixas (bootstrap independente por faixa; IC95% que exclui 0 = diferença sustentada):")
    por_faixa = {b: g for b, g in df.groupby("skin_tone_band")}
    for a, b in [("claro (FST I-II)", "escuro (FST V-VI)"), ("médio (FST III-IV)", "escuro (FST V-VI)"),
                 ("claro (FST I-II)", "médio (FST III-IV)")]:
        diffs = []
        for _ in range(N_BOOT):
            ia = rng.integers(0, len(por_faixa[a]), len(por_faixa[a]))
            ib = rng.integers(0, len(por_faixa[b]), len(por_faixa[b]))
            ya, sa = por_faixa[a]["y"].to_numpy()[ia], por_faixa[a]["score_mel"].to_numpy()[ia]
            yb, sb = por_faixa[b]["y"].to_numpy()[ib], por_faixa[b]["score_mel"].to_numpy()[ib]
            if ya.min() == ya.max() or yb.min() == yb.max():
                continue
            diffs.append(roc_auc_score(ya, sa) - roc_auc_score(yb, sb))
        diffs = np.array(diffs)
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        print(f"  {a} - {b}: {diffs.mean():+.3f} [IC95% {np.percentile(diffs, 2.5):+.3f} a {np.percentile(diffs, 97.5):+.3f}] p(bootstrap)~{p:.3f}")

    print("\nComposição de doenças malignas por faixa (a mistura de doenças pode confundir a comparação):")
    print(pd.crosstab(df[df["y"] == 1]["disease"], df[df["y"] == 1]["skin_tone_band"]).sort_values(
        by="escuro (FST V-VI)", ascending=False).head(8))

    mel = df[df["disease"].fillna("").str.contains("melanoma", case=False)]
    print(f"\nSó os {len(mel)} casos de melanoma do DDI (observação qualitativa, n pequeno demais p/ métrica robusta):")
    print(mel.groupby("skin_tone_band")["score_mel"].agg(n="count", mediana="median",
          detectados_no_limiar=lambda s: int((s >= thr).sum())).round(3))

    print(f"\nPrevisões por imagem: {PRED_PATH}\nResumo: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
