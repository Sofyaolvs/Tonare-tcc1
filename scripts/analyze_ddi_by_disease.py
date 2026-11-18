"""
Sprint 3 — Análise do DDI controlando pela mistura de doenças.

Motivo: no DDI as faixas de tom de pele têm doenças malignas muito diferentes (ex: faixa escura tem
26 de 48 malignos de micose fungoide e 0 basocelular). Comparar a faixa inteira mistura tom de pele
com tipo de doença. Aqui a comparação é feita DENTRO de cada grupo de doença, e há uma sensibilidade
padronizada (mesmo peso de doenças em todas as faixas) usando só os grupos presentes nas 3 faixas.

Usa as previsões geradas por scripts/evaluate_ddi.py (data/processed/ddi/ddi_predictions.csv).

Uso:
    python scripts/evaluate_ddi.py && python scripts/analyze_ddi_by_disease.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "data" / "processed" / "ddi" / "ddi_predictions.csv"
# v2 = referência (recomendada, DDI interpretável); v3 = achado sobre generalização cross-modalidade
CHECKPOINT_VERSION = "v2"
CHECKPOINT = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{CHECKPOINT_VERSION}.pt"
OUT_PATH = BASE_DIR / "data" / "processed" / "ddi" / "ddi_by_disease_summary.csv"
N_BOOT = 2000
SEED = 42
BANDS = ["claro", "médio", "escuro"]
COMUNS = ["melanoma", "carcinoma espinocelular"]  # grupos malignos presentes nas 3 faixas com n razoável


def grupo_doenca(disease: str) -> str:
    if "melanoma" in disease:
        return "melanoma"
    if "basal" in disease:
        return "carcinoma basocelular"
    if "squamous" in disease:
        return "carcinoma espinocelular"
    if "mycosis" in disease or "lymphoma" in disease:
        return "linfoma cutâneo"
    return "outros"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    meia = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centro - meia, centro + meia


def main() -> None:
    thr = float(torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["threshold"])
    d = pd.read_csv(PRED_PATH)
    d["band"] = d["skin_tone_band"].str.split(" ").str[0]
    d["grupo"] = np.where(d["malignant"], d["disease"].map(grupo_doenca), "benigno")
    rng = np.random.default_rng(SEED)
    print(f"Limiar do checkpoint: {thr:.2f}\n")

    benignos = d[d["grupo"] == "benigno"]
    print("1) Benignos (falsos positivos) por faixa — o comportamento nos casos benignos difere entre tons?")
    for b in BANDS:
        g = benignos[benignos["band"] == b]
        fp = int((g["score_mel"] >= thr).sum())
        lo, hi = wilson(fp, len(g))
        print(f"   {b:<7} n={len(g):>3}  marcados como maligno: {fp/len(g):.3f} [IC95% {lo:.3f}-{hi:.3f}]  escore médio={g['score_mel'].mean():.3f}")

    print("\n2) Sensibilidade no limiar, DENTRO de cada grupo de doença maligna (IC95% de Wilson):")
    linhas = []
    for grp in ["melanoma", "carcinoma espinocelular", "carcinoma basocelular", "linfoma cutâneo", "outros"]:
        partes = []
        for b in BANDS:
            g = d[(d["grupo"] == grp) & (d["band"] == b)]
            k = int((g["score_mel"] >= thr).sum())
            lo, hi = wilson(k, len(g))
            linhas.append({"grupo": grp, "faixa": b, "n": len(g), "detectados": k,
                           "sens": k / len(g) if len(g) else np.nan, "ic_lo": lo, "ic_hi": hi,
                           "escore_mediano": g["score_mel"].median() if len(g) else np.nan})
            partes.append(f"{b}: {k}/{len(g)}" + (f" ({k/len(g):.2f} [{lo:.2f}-{hi:.2f}])" if len(g) else ""))
        print(f"   {grp:<26} " + " | ".join(partes))
    pd.DataFrame(linhas).to_csv(OUT_PATH, index=False)

    print(f"\n3) Sensibilidade PADRONIZADA (só {', '.join(COMUNS)}; mesmo peso de doenças em todas as faixas):")
    pos = d[d["grupo"].isin(COMUNS)]
    pesos = pos["grupo"].value_counts(normalize=True).to_dict()
    print("   pesos (mistura conjunta):", {k: round(v, 2) for k, v in pesos.items()})

    def sens_padr(sub: pd.DataFrame) -> float:
        tot = 0.0
        for grp, w in pesos.items():
            g = sub[sub["grupo"] == grp]
            if len(g):
                tot += w * (g["score_mel"] >= thr).mean()
        return tot

    boot = {b: [] for b in BANDS}
    for b in BANDS:
        sub = pos[pos["band"] == b]
        obs = sens_padr(sub)
        for _ in range(N_BOOT):
            reamostra = pd.concat([sub[sub["grupo"] == grp].sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
                                   for grp in pesos])
            boot[b].append(sens_padr(reamostra))
        lo, hi = np.percentile(boot[b], [2.5, 97.5])
        print(f"   {b:<7} n={len(sub):>3}  sensibilidade padronizada = {obs:.3f} [IC95% {lo:.3f}-{hi:.3f}]")
    for a, b in [("claro", "escuro"), ("médio", "escuro"), ("claro", "médio")]:
        diff = np.array(boot[a]) - np.array(boot[b])
        print(f"   {a} - {b}: {diff.mean():+.3f} [IC95% {np.percentile(diff, 2.5):+.3f} a {np.percentile(diff, 97.5):+.3f}]")

    print("\n4) AUC (grupo maligno vs TODOS os benignos da mesma faixa), só onde há >= 7 casos:")
    for grp in ["melanoma", "carcinoma espinocelular", "linfoma cutâneo"]:
        partes = []
        for b in BANDS:
            p = d[(d["grupo"] == grp) & (d["band"] == b)]
            n_ = d[(d["grupo"] == "benigno") & (d["band"] == b)]
            if len(p) < 7:
                partes.append(f"{b}: n={len(p)} (pouco)")
                continue
            y = np.r_[np.ones(len(p)), np.zeros(len(n_))]
            s = np.r_[p["score_mel"].to_numpy(), n_["score_mel"].to_numpy()]
            aucs = []
            for _ in range(N_BOOT):
                ip = rng.integers(0, len(p), len(p))
                inn = rng.integers(0, len(n_), len(n_))
                aucs.append(roc_auc_score(np.r_[np.ones(len(ip)), np.zeros(len(inn))],
                                          np.r_[p["score_mel"].to_numpy()[ip], n_["score_mel"].to_numpy()[inn]]))
            lo, hi = np.percentile(aucs, [2.5, 97.5])
            partes.append(f"{b}: {roc_auc_score(y, s):.2f} [{lo:.2f}-{hi:.2f}] (n={len(p)})")
        print(f"   {grp:<26} " + " | ".join(partes))

    print(f"\nResumo salvo em: {OUT_PATH}")


if __name__ == "__main__":
    main()
