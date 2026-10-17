"""
Gera os DADOS e os GRÁFICOS da documentação (pasta docs/) a partir das previsões salvas de cada
versão do modelo (v2 a v7). Nada aqui treina modelo: só pontua o teste do HAM10000 (se ainda não
estiver em cache), junta as previsões dos testes externos e calcula métricas com IC95% (bootstrap por
lesão), comparações pareadas entre versões, análise por tom de pele e figuras.

Pré-requisitos: as avaliações de cada versão já rodadas (evaluate_milk10k.py, evaluate_padufes20.py,
evaluate_external.py, evaluate_ddi.py, evaluate_fitzpatrick17k.py).

Saídas:
    docs/data/*.csv        tabelas de resultados
    docs/figures/*.png     gráficos

Uso:
    python scripts/build_docs_report.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import models, transforms

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
PROC = Path(os.environ.get("TONARE_PROC", BASE / "data" / "processed"))   # no Kaggle: pasta do dataset de entradas
DOCS = Path(os.environ.get("TONARE_DOCS", BASE / "docs"))
FIG, DATA = DOCS / "figures", DOCS / "data"
VERSIONS = ["v2", "v3", "v4", "v5", "v6", "v7"]
N_BOOT, SEED = 1000, 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TESTS = ["HAM10000 (dermatoscopia)", "MILK10k reservado, dermatoscopia", "MILK10k reservado, foto clínica",
         "PAD-UFES-20 (foto de celular)", "HIBA, dermatoscopia", "HIBA, foto clínica",
         "MSK 2020 (população real)", "DDI (foto clínica)"]
COLORS = dict(zip(VERSIONS, plt.cm.viridis(np.linspace(0.05, 0.92, len(VERSIONS)))))

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def load_ckpt(v):
    ck = torch.load(PROC / f"model_resnet50_baseline_{v}.pt", map_location="cpu", weights_only=True)
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, float(ck["threshold"])


def ham_scores(v) -> pd.DataFrame:
    cache = PROC / f"ham_test_scores_{v}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    d = pd.read_csv(PROC / "split_index.csv")
    d = d[(d["split"] == "test") & (~d["is_augmented"])].reset_index(drop=True)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    model, _ = load_ckpt(v)
    paths = [BASE / p.replace("\\", "/") for p in d["path"]]
    s = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            s += torch.softmax(model(torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 64]])), 1)[:, 1].tolist()
    out = pd.DataFrame({"key": d["image_id"], "y": (d["dx"] == "mel").astype(int), "grp": d["lesion_id"], "score": s})
    out.to_csv(cache, index=False)
    return out


def load_test(name: str, v: str) -> pd.DataFrame:
    """key, y, grp, score (+ colunas extras de tom quando existem)."""
    if name.startswith("HAM10000"):
        return ham_scores(v)
    if name.startswith("MILK10k"):
        d = pd.read_csv(PROC / "milk10k" / f"milk10k_predictions_{v}_heldout.csv")
        d = d[d["modalidade"] == ("dermatoscopia" if "dermatoscopia" in name else "foto clínica")]
        return pd.DataFrame({"key": d["isic_id"], "y": d["y"], "grp": d["lesion_id"], "score": d["score_mel"], "tone": d["skin_tone_class"]})
    if name.startswith("PAD-UFES"):
        d = pd.read_csv(PROC / "padufes20" / f"padufes20_predictions_{v}.csv")
        return pd.DataFrame({"key": d["img_id"], "y": d["y"], "grp": d["lesion_id"], "score": d["score_mel"]})
    if name.startswith("HIBA"):
        d = pd.read_csv(PROC / "external" / f"hiba251_{v}.csv")
        d = d[d["modalidade"] == ("dermatoscopia" if "dermatoscopia" in name else "foto clínica")]
        return pd.DataFrame({"key": d["isic_id"], "y": d["y"], "grp": d["grp"], "score": d["score_mel"]})
    if name.startswith("MSK"):
        d = pd.read_csv(PROC / "external" / f"consec_mel2020_216_{v}.csv")
        return pd.DataFrame({"key": d["isic_id"], "y": d["y"], "grp": d["grp"], "score": d["score_mel"]})
    if name.startswith("DDI"):
        d = pd.read_csv(PROC / "ddi" / ("ddi_predictions.csv" if v == "v2" else f"ddi_predictions_{v}.csv"))
        return pd.DataFrame({"key": d["ddi_file"], "y": d["y"], "grp": d["ddi_file"], "score": d["score_mel"], "band": d["skin_tone_band"]})
    raise KeyError(name)


def wide(name: str) -> pd.DataFrame:
    """Um dataframe por teste com a pontuação de cada versão alinhada pela mesma imagem."""
    base = None
    for v in VERSIONS:
        d = load_test(name, v)
        d = d.set_index("key")
        if base is None:
            base = d.drop(columns="score").copy()
        base[f"s_{v}"] = d["score"].reindex(base.index)
    return base.dropna(subset=[f"s_{v}" for v in VERSIONS]).reset_index()


def boot_groups(df: pd.DataFrame):
    return list(df.reset_index(drop=True).groupby("grp").indices.values())


def auc_ci(y, s, groups, rng):
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].min() != y[idx].max():
            vals.append(roc_auc_score(y[idx], s[idx]))
    return float(roc_auc_score(y, s)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    def get_thr(v):  # sem o .pt (Kaggle), lê o limiar de thresholds.json
        return load_ckpt(v)[1] if (PROC / f"model_resnet50_baseline_{v}.pt").exists() else json.loads((PROC / "thresholds.json").read_text())[v]
    thr = {v: get_thr(v) for v in VERSIONS}

    # ---------- 1) métricas por versão x teste ----------
    W = {}
    rows = []
    for t in TESTS:
        W[t] = wide(t)
        d = W[t]
        y, g = d["y"].to_numpy(), boot_groups(d)
        for v in VERSIONS:
            s = d[f"s_{v}"].to_numpy()
            a, lo, hi = auc_ci(y, s, g, rng)
            p = s >= thr[v]
            se, sp = float(p[y == 1].mean()), float((~p[y == 0]).mean())
            rows.append({"teste": t, "versao": v, "n_mel": int(y.sum()), "n_nv": int((y == 0).sum()), "auc": a, "auc_lo": lo, "auc_hi": hi,
                         "limiar": thr[v], "sensibilidade": se, "especificidade": sp, "acuracia": float((p == y).mean()),
                         "acuracia_balanceada": (se + sp) / 2, "acuracia_sempre_benigno": float((y == 0).mean())})
    R = pd.DataFrame(rows)
    R.to_csv(DATA / "resultados_por_versao_e_teste.csv", index=False)

    # Fitzpatrick17k V-VI (na v7 a fonte do treino é a mesma -> separado, com aviso)
    fz = {v: pd.read_csv(PROC / "fitzpatrick17k" / f"fitz17k_{v}.csv") for v in VERSIONS}
    fzb = fz["v2"][(fz["v2"]["grupo"] == "V-VI") & (fz["v2"]["label"] != "lentigo maligna")][["md5hash", "y"]].copy()
    for v in VERSIONS:
        fzb[f"s_{v}"] = fzb["md5hash"].map(fz[v].set_index("md5hash")["score_mel"])
    fzb["grp"] = fzb["md5hash"]
    fz_rows = []
    for v in VERSIONS:
        y, s = fzb["y"].to_numpy(), fzb[f"s_{v}"].to_numpy()
        a, lo, hi = auc_ci(y, s, boot_groups(fzb), rng)
        p = s >= thr[v]
        fz_rows.append({"versao": v, "n_mel": int(y.sum()), "n_nv": int((y == 0).sum()), "auc": a, "auc_lo": lo, "auc_hi": hi,
                        "sensibilidade": float(p[y == 1].mean()), "especificidade": float((~p[y == 0]).mean()),
                        "mesma_fonte_do_treino": v == "v7"})
    pd.DataFrame(fz_rows).to_csv(DATA / "fitzpatrick17k_tons_V_VI.csv", index=False)

    # ---------- 2) comparação pareada ----------
    pairs = [("v3", "v2"), ("v4", "v2"), ("v5", "v4"), ("v6", "v5"), ("v7", "v6"), ("v7", "v2")]
    prow = []
    for t in TESTS:
        d = W[t]
        y, g = d["y"].to_numpy(), boot_groups(d)
        for a_, b_ in pairs:
            sa, sb = d[f"s_{a_}"].to_numpy(), d[f"s_{b_}"].to_numpy()
            diffs = []
            for _ in range(N_BOOT):
                idx = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
                if y[idx].min() != y[idx].max():
                    diffs.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
            diffs = np.array(diffs)
            prow.append({"teste": t, "comparacao": f"{a_} - {b_}", "diff_auc": float(roc_auc_score(y, sa) - roc_auc_score(y, sb)),
                         "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
                         "prob_primeira_melhor": float((diffs > 0).mean())})
    PR = pd.DataFrame(prow)
    PR.to_csv(DATA / "comparacao_pareada_auc.csv", index=False)

    # ---------- 3) tom de pele ----------
    frows = []
    tone_map = lambda c: "escuro (0-2)" if c <= 2 else ("médio (3)" if c == 3 else "claro (4-5)")
    for t in ("MILK10k reservado, dermatoscopia", "MILK10k reservado, foto clínica"):
        d = W[t].copy()
        d["grupo"] = d["tone"].map(tone_map)
        for gname, sub in d.groupby("grupo"):
            y, gg = sub["y"].to_numpy(), boot_groups(sub)
            for v in VERSIONS:
                a, lo, hi = auc_ci(y, sub[f"s_{v}"].to_numpy(), gg, rng)
                p = sub[f"s_{v}"].to_numpy() >= thr[v]
                frows.append({"teste": t, "grupo": gname, "versao": v, "n_mel": int(y.sum()), "n_nv": int((y == 0).sum()), "auc": a, "auc_lo": lo,
                              "auc_hi": hi, "sensibilidade": float(p[y == 1].mean()), "especificidade": float((~p[y == 0]).mean())})
    d = W["DDI (foto clínica)"]
    for gname, sub in d.groupby("band"):
        y, gg = sub["y"].to_numpy(), boot_groups(sub)
        for v in VERSIONS:
            a, lo, hi = auc_ci(y, sub[f"s_{v}"].to_numpy(), gg, rng)
            p = sub[f"s_{v}"].to_numpy() >= thr[v]
            frows.append({"teste": "DDI (foto clínica)", "grupo": gname, "versao": v, "n_mel": int(y.sum()), "n_nv": int((y == 0).sum()), "auc": a,
                          "auc_lo": lo, "auc_hi": hi, "sensibilidade": float(p[y == 1].mean()), "especificidade": float((~p[y == 0]).mean())})
    FR = pd.DataFrame(frows)
    FR.to_csv(DATA / "fairness_por_tom_de_pele.csv", index=False)

    # ---------- 4) composição do treino por versão ----------
    def comp(path, v):
        m = pd.read_csv(PROC / path)
        if "dataset" not in m:
            m["dataset"] = "ham10000"
        if "image_type" not in m:
            m["image_type"] = "dermoscopic"
        m = m[(m["split"] == "train") & (~m["is_augmented"])]
        m["fonte"] = m["dataset"] + " (" + m["image_type"].map({"dermoscopic": "derm.", "clinical": "clínica"}) + ")"
        eff = m.groupby(["fonte", "dx"]).size().unstack(fill_value=0)
        dist = m.drop_duplicates("path").groupby(["fonte", "dx"]).size().unstack(fill_value=0)
        out = []
        for f_ in eff.index:
            out.append({"versao": v, "fonte": f_, "imagens_efetivas_por_epoca": int(eff.loc[f_].sum()), "imagens_distintas": int(dist.loc[f_].sum()),
                        "melanoma_efetivo": int(eff.loc[f_].get("mel", 0)), "nevo_efetivo": int(eff.loc[f_].get("nv", 0))})
        return out
    comp_rows = []
    for v, path in [("v2", "split_index.csv"), ("v3", "split_index_combined.csv"), ("v4", "split_index_v4.csv"), ("v5", "split_index_v5.csv"),
                    ("v6", "split_index_v6.csv"), ("v7", "split_index_v7.csv")]:
        comp_rows += comp(path, v)
    C = pd.DataFrame(comp_rows)
    C.to_csv(DATA / "composicao_do_treino_por_versao.csv", index=False)

    # =============== FIGURAS ===============
    short = {t: t.replace(" (dermatoscopia)", "\n(derm.)").replace("MILK10k reservado, ", "MILK10k\nreserv., ").replace("PAD-UFES-20 (foto de celular)", "PAD-UFES-20\n(celular)")
             .replace("HIBA, ", "HIBA, ").replace("MSK 2020 (população real)", "MSK 2020\n(pop. real)").replace("DDI (foto clínica)", "DDI\n(clínica)") for t in TESTS}

    # fig01 heatmap
    M = R.pivot(index="teste", columns="versao", values="auc").loc[TESTS, VERSIONS]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(M.values, cmap="YlGnBu", vmin=0.6, vmax=0.95, aspect="auto")
    ax.set_xticks(range(len(VERSIONS)), VERSIONS)
    ax.set_yticks(range(len(TESTS)), TESTS)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=9, color="white" if M.values[i, j] > 0.82 else "black")
    ax.set_title("AUC por versão do modelo e por teste (nenhum teste entra no treino)")
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout(); fig.savefig(FIG / "fig01_auc_heatmap.png"); plt.close(fig)

    # fig02 barras com IC
    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.13
    for j, v in enumerate(VERSIONS):
        sub = R[R["versao"] == v].set_index("teste").loc[TESTS]
        x = np.arange(len(TESTS)) + (j - 2.5) * w
        ax.bar(x, sub["auc"], w, color=COLORS[v], label=v, yerr=[sub["auc"] - sub["auc_lo"], sub["auc_hi"] - sub["auc"]], error_kw={"lw": 0.8, "capsize": 1.5})
    ax.set_xticks(range(len(TESTS)), [short[t] for t in TESTS], fontsize=8.5)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC (IC 95%)"); ax.legend(ncol=6, frameon=False, loc="upper right")
    ax.set_title("Comparação das versões v2 a v7 em cada teste")
    fig.tight_layout(); fig.savefig(FIG / "fig02_auc_barras_por_teste.png"); plt.close(fig)

    # fig03 lacuna de modalidade
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (nome, a_, b_) in zip(axs, [("MILK10k reservado", "MILK10k reservado, dermatoscopia", "MILK10k reservado, foto clínica"), ("HIBA", "HIBA, dermatoscopia", "HIBA, foto clínica")]):
        ra = R[R["teste"] == a_].set_index("versao").loc[VERSIONS]; rb = R[R["teste"] == b_].set_index("versao").loc[VERSIONS]
        ax.errorbar(range(6), ra["auc"], yerr=[ra["auc"] - ra["auc_lo"], ra["auc_hi"] - ra["auc"]], marker="o", capsize=2, label="dermatoscopia", color="#1f6f8b")
        ax.errorbar(np.arange(6) + 0.08, rb["auc"], yerr=[rb["auc"] - rb["auc_lo"], rb["auc_hi"] - rb["auc"]], marker="s", capsize=2, label="foto clínica", color="#d1495b")
        ax.set_xticks(range(6), VERSIONS); ax.set_title(nome); ax.set_ylim(0.6, 1.0)
    axs[0].set_ylabel("AUC (IC 95%)"); axs[0].legend(frameon=False)
    fig.suptitle("A foto clínica continua mais difícil que a dermatoscopia em todas as versões", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig03_lacuna_de_modalidade.png"); plt.close(fig)

    # fig04 ROC v7 por teste
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for t, c in zip(TESTS, plt.cm.tab10(np.linspace(0, 1, 10))):
        d = W[t]; fpr, tpr, _ = roc_curve(d["y"], d["s_v7"])
        ax.plot(fpr, tpr, color=c, lw=1.6, label=f"{t} (AUC {R[(R.teste == t) & (R.versao == 'v7')].auc.iloc[0]:.2f})")
        p = d["s_v7"].to_numpy() >= thr["v7"]; ax.scatter((p[d["y"] == 0]).mean(), (p[d["y"] == 1]).mean(), color=c, s=28, zorder=5, edgecolor="black", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("1 - especificidade (taxa de falso positivo)"); ax.set_ylabel("sensibilidade")
    ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.set_title("Curvas ROC da v7 (pontos = limiar 0,15)")
    fig.tight_layout(); fig.savefig(FIG / "fig04_roc_v7_por_teste.png"); plt.close(fig)

    # fig05 ROC v2 x v7 em 4 testes
    key4 = ["HAM10000 (dermatoscopia)", "MSK 2020 (população real)", "PAD-UFES-20 (foto de celular)", "MILK10k reservado, foto clínica"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, t in zip(axs, key4):
        d = W[t]
        for v in ("v2", "v5", "v7"):
            fpr, tpr, _ = roc_curve(d["y"], d[f"s_{v}"]); ax.plot(fpr, tpr, color=COLORS[v], lw=1.7, label=f"{v} ({R[(R.teste == t) & (R.versao == v)].auc.iloc[0]:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7); ax.set_title(t, fontsize=9); ax.set_xlabel("1 - especificidade"); ax.legend(frameon=False, fontsize=8, loc="lower right", title="AUC", title_fontsize=8)
    axs[0].set_ylabel("sensibilidade"); fig.tight_layout(); fig.savefig(FIG / "fig05_roc_v2_v5_v7.png"); plt.close(fig)

    # fig06 trade-off de limiar (HAM, v7)
    d = W["HAM10000 (dermatoscopia)"]; y, s = d["y"].to_numpy(), d["s_v7"].to_numpy(); ts = np.linspace(0.01, 0.99, 99)
    se = [(s[y == 1] >= t).mean() for t in ts]; sp = [(s[y == 0] < t).mean() for t in ts]; ac = [((s >= t) == y).mean() for t in ts]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ts, se, label="sensibilidade", color="#d1495b"); ax.plot(ts, sp, label="especificidade", color="#1f6f8b"); ax.plot(ts, ac, label="acurácia", color="#444", ls="--")
    ax.axhline((y == 0).mean(), color="gray", ls=":", label=f"acurácia de 'sempre benigno' ({(y == 0).mean():.1%})")
    ax.axvline(thr["v7"], color="black", lw=1); ax.text(thr["v7"] + 0.01, 0.05, "limiar do projeto (0,15)", fontsize=8)
    ax.set_xlabel("limiar de decisão"); ax.set_ylabel("valor"); ax.set_title("O limiar troca sensibilidade por especificidade (v7, teste do HAM10000)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig06_tradeoff_do_limiar_v7.png"); plt.close(fig)

    # fig07 acurácia x prevalência
    se7, sp7 = R[(R.teste == TESTS[0]) & (R.versao == "v7")][["sensibilidade", "especificidade"]].iloc[0]
    prev = np.linspace(0.005, 0.5, 100)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(prev * 100, (se7 * prev + sp7 * (1 - prev)) * 100, label="v7 no limiar 0,15", color="#1f6f8b")
    ax.plot(prev * 100, (1 - prev) * 100, label="sempre responder 'benigno'", color="#d1495b", ls="--")
    ax.set_xlabel("% de melanomas na população"); ax.set_ylabel("acurácia (%)"); ax.legend(frameon=False)
    ax.set_title("Quanto mais raro o melanoma, mais a acurácia engana")
    fig.tight_layout(); fig.savefig(FIG / "fig07_acuracia_vs_prevalencia.png"); plt.close(fig)

    # fig08 composição do treino
    piv = C.pivot_table(index="versao", columns="fonte", values="imagens_efetivas_por_epoca", aggfunc="sum", fill_value=0).loc[VERSIONS]
    fig, ax = plt.subplots(figsize=(9, 4.8)); bottom = np.zeros(len(VERSIONS))
    for col, c in zip(piv.columns, plt.cm.Set2(np.linspace(0, 1, len(piv.columns)))):
        ax.bar(VERSIONS, piv[col], bottom=bottom, label=col, color=c); bottom += piv[col].to_numpy()
    ax.set_ylabel("imagens vistas por época"); ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left"); ax.set_title("Composição do treino por versão")
    fig.tight_layout(); fig.savefig(FIG / "fig08_composicao_do_treino.png"); plt.close(fig)

    # fig09 tom de pele (v6 x v7, IC)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    orders = {"MILK10k reservado, dermatoscopia": ["escuro (0-2)", "médio (3)", "claro (4-5)"], "MILK10k reservado, foto clínica": ["escuro (0-2)", "médio (3)", "claro (4-5)"],
              "DDI (foto clínica)": ["escuro (FST V-VI)", "médio (FST III-IV)", "claro (FST I-II)"]}
    for ax, (t, order) in zip(axs, orders.items()):
        for k, v in enumerate(("v6", "v7")):
            sub = FR[(FR["teste"] == t) & (FR["versao"] == v)].set_index("grupo").loc[order]
            ax.errorbar(np.arange(3) + (k - 0.5) * 0.16, sub["auc"], yerr=[sub["auc"] - sub["auc_lo"], sub["auc_hi"] - sub["auc"]], fmt="o", capsize=3, color=COLORS[v], label=v)
        ax.set_xticks(range(3), [o.replace(" (", "\n(") for o in order], fontsize=8); ax.set_title(t, fontsize=9)
    axs[0].set_ylabel("AUC (IC 95%)"); axs[0].legend(frameon=False)
    fig.suptitle("AUC por tom de pele: os intervalos se sobrepõem (poucos casos nas faixas escuras)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig09_auc_por_tom_de_pele.png"); plt.close(fig)

    # fig10 diferença pareada v7 - v6
    sub = PR[PR["comparacao"] == "v7 - v6"].set_index("teste").loc[TESTS[::-1]]
    fig, ax = plt.subplots(figsize=(7.8, 4.6)); yy = np.arange(len(sub))
    ax.errorbar(sub["diff_auc"], yy, xerr=[sub["diff_auc"] - sub["lo"], sub["hi"] - sub["diff_auc"]], fmt="o", color="#1f6f8b", capsize=3)
    ax.axvline(0, color="black", lw=0.9); ax.set_yticks(yy, sub.index, fontsize=8.5); ax.set_xlabel("diferença de AUC (v7 - v6), IC 95% pareado")
    ax.set_title("v7 contra v6: só o HAM10000 exclui o zero")
    fig.tight_layout(); fig.savefig(FIG / "fig10_diferenca_pareada_v7_v6.png"); plt.close(fig)

    # fig11 pontos operacionais
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.4), sharex=True, sharey=True)
    for ax, t in zip(axs.ravel(), TESTS):
        sub = R[R["teste"] == t].set_index("versao").loc[VERSIONS]
        for v in VERSIONS:
            ax.scatter(1 - sub.loc[v, "especificidade"], sub.loc[v, "sensibilidade"], color=COLORS[v], s=55, edgecolor="black", linewidth=0.5, zorder=3)
            ax.annotate(v, (1 - sub.loc[v, "especificidade"], sub.loc[v, "sensibilidade"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k:", lw=0.7); ax.set_title(t, fontsize=8.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for ax in axs[1]: ax.set_xlabel("1 - especificidade")
    for ax in axs[:, 0]: ax.set_ylabel("sensibilidade")
    fig.suptitle("Ponto operacional de cada versão no seu limiar salvo (canto superior esquerdo é melhor)", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig11_pontos_operacionais.png"); plt.close(fig)

    # fig12 escassez de melanoma por tom (dados da auditoria; ver docs/02-datasets.md)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    cats = ["Fitzpatrick17k (3 rótulos)", "DDI (biópsia)", "HIBA", "PAD-UFES-20"]
    clar = [290, 7, 224, 38]; med = [132, 7, 19, 14]; esc = [43, 7, 0, 0]
    x = np.arange(len(cats))
    ax.bar(x - 0.27, clar, 0.27, label="I-II", color="#f2c14e"); ax.bar(x, med, 0.27, label="III-IV", color="#d1913c")
    ax.bar(x + 0.27, esc, 0.27, label="V-VI", color="#5a3825")
    for xi, v in zip(x + 0.27, esc): ax.text(xi, v + 4, str(v), ha="center", fontsize=8)
    ax.set_xticks(x, cats, fontsize=8.5); ax.set_ylabel("melanomas"); ax.legend(frameon=False, title="Fitzpatrick"); ax.set_title("Datasets com escala Fitzpatrick", fontsize=10)
    lv = ["0-1 (muito escuro)", "2", "3", "4", "5 (muito claro)"]; mel_lv = [16, 121, 200, 90, 23]
    ax2.bar(range(5), mel_lv, color=["#5a3825", "#8c5a3a", "#d1913c", "#e5b25d", "#f2c14e"])
    for xi, v in enumerate(mel_lv): ax2.text(xi, v + 4, str(v), ha="center", fontsize=8)
    ax2.set_xticks(range(5), lv, fontsize=8.5); ax2.set_title("MILK10k (escala própria 0-5, lesões)", fontsize=10)
    fig.suptitle("Melanoma por tom de pele nos datasets auditados: os tons realmente escuros quase não existem", y=1.0)
    fig.tight_layout(); fig.savefig(FIG / "fig12_melanoma_por_tom_nos_datasets.png"); plt.close(fig)

    # fig13 curva de treino da v7 (log do Kaggle)
    tr_auc = [0.833, 0.854, 0.859, 0.861, 0.863, 0.875, 0.889, 0.902, 0.910, 0.923, 0.930, 0.938, 0.945, 0.951, 0.958, 0.962, 0.968]
    va_auc = [0.849, 0.849, 0.855, 0.852, 0.851, 0.872, 0.879, 0.884, 0.886, 0.890, 0.892, 0.890, 0.893, 0.894, 0.896, 0.891, 0.898]
    tr_loss = [0.503, 0.474, 0.466, 0.463, 0.460, 0.440, 0.415, 0.391, 0.376, 0.352, 0.335, 0.317, 0.298, 0.283, 0.263, 0.250, 0.232]
    va_loss = [0.452, 0.464, 0.441, 0.457, 0.470, 0.415, 0.402, 0.396, 0.396, 0.394, 0.394, 0.399, 0.402, 0.406, 0.403, 0.416, 0.414]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4)); ep = np.arange(1, 18)
    for ax, a, b, nome in [(axs[0], tr_auc, va_auc, "AUC"), (axs[1], tr_loss, va_loss, "loss")]:
        ax.plot(ep, a, marker="o", ms=3, label="treino", color="#1f6f8b"); ax.plot(ep, b, marker="o", ms=3, label="validação", color="#d1495b")
        ax.axvline(5.5, color="gray", ls=":"); ax.set_xlabel("época (1-5: só camada final; 6-17: rede inteira)"); ax.set_ylabel(nome); ax.legend(frameon=False)
    axs[0].set_title("AUC: a validação estabiliza em ~0,90 enquanto o treino sobe (leve overfitting)"); axs[1].set_title("loss")
    fig.tight_layout(); fig.savefig(FIG / "fig13_curva_de_treino_v7.png"); plt.close(fig)

    print("OK. Arquivos em", DOCS)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n== AUC por versão x teste =="); print(R.pivot(index="teste", columns="versao", values="auc").loc[TESTS, VERSIONS].round(3).to_string())
    print("\n== Fitz V-VI =="); print(pd.DataFrame(fz_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
