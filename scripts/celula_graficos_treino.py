# ===== GRÁFICOS E MÉTRICAS DESTE TREINO (célula única: serve para qualquer versão) =====
# Rode DEPOIS da célula que escolhe o limiar e avalia o teste (precisa de val_labels, val_probs,
# test_labels, test_probs, best_threshold, history_stage1/2). Salva tudo em /kaggle/working.
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

NOME_VERSAO = None          # ex.: "v7". Se None, tenta deduzir do nome do checkpoint salvo
SAIDA_BASE = "/kaggle/working"

if NOME_VERSAO is None:
    try:
        NOME_VERSAO = Path(checkpoint_path).stem.split("_")[-1]
    except NameError:
        NOME_VERSAO = "treino"
OUT = Path(SAIDA_BASE) / f"graficos_{NOME_VERSAO}"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
AZUL, VERMELHO, CINZA = "#1f6f8b", "#d1495b", "#555555"

yv, pv = np.asarray(val_labels), np.asarray(val_probs)
yt, pt = np.asarray(test_labels), np.asarray(test_probs)
thr = float(best_threshold)
hist1 = globals().get("history_stage1", []) or []
hist2 = globals().get("history_stage2", []) or []


def ponto(y, p, t):
    pred = p >= t
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    fp, tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else float("nan")
    return {"limiar": float(t), "sensibilidade": div(tp, tp + fn), "especificidade": div(tn, tn + fp), "precisao": div(tp, tp + fp),
            "acuracia": (tp + tn) / len(y), "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(OUT / nome)
    plt.show()
    plt.close(fig)


# 1) curvas de treino -------------------------------------------------------------------
if hist1 or hist2:
    ep = hist1 + hist2
    x = np.arange(1, len(ep) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, chave, nome in [(axs[0], "auc", "AUC"), (axs[1], "loss", "loss")]:
        ax.plot(x, [h["train"][chave] for h in ep], marker="o", ms=3, color=AZUL, label="treino")
        ax.plot(x, [h["val"][chave] for h in ep], marker="o", ms=3, color=VERMELHO, label="validação")
        if hist1 and hist2:
            ax.axvline(len(hist1) + 0.5, color="gray", ls=":")
        for off, h in ((0, hist1), (len(hist1), hist2)):
            if h:
                k = int(np.argmax([e["val"]["auc"] for e in h]))
                ax.scatter(off + k + 1, h[k]["val"][chave], marker="*", s=130, color="gold", edgecolor="black", zorder=5)
        ax.set_xlabel("época (estrela = época restaurada pelo early stopping)"); ax.set_ylabel(nome); ax.legend(frameon=False)
    axs[0].set_title("AUC por época"); axs[1].set_title("loss por época")
    salvar(fig, "01_curvas_de_treino.png")

# 2) ROC ----------------------------------------------------------------------------------
rng = np.random.default_rng(42)
def auc_ic(y, p, n=500):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

auc_v, auc_t = roc_auc_score(yv, pv), roc_auc_score(yt, pt)
ic_t = auc_ic(yt, pt)
fig, ax = plt.subplots(figsize=(6, 5.6))
for y, p, cor, nome in [(yv, pv, VERMELHO, f"validação (AUC {auc_v:.3f})"), (yt, pt, AZUL, f"teste (AUC {auc_t:.3f} [{ic_t[0]:.3f}-{ic_t[1]:.3f}])")]:
    f, t, _ = roc_curve(y, p); ax.plot(f, t, color=cor, lw=1.8, label=nome)
for t_, mk, nome in [(thr, "o", f"limiar {thr:.2f}"), (0.5, "s", "limiar 0,50")]:
    q = ponto(yt, pt, t_); ax.scatter(1 - q["especificidade"], q["sensibilidade"], marker=mk, s=60, color="black", zorder=5, label=f"teste, {nome}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("1 - especificidade"); ax.set_ylabel("sensibilidade"); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title(f"Curva ROC ({NOME_VERSAO}); IC do teste por bootstrap de imagem (otimista)")
salvar(fig, "02_curva_roc.png")

# 3) precisão x revocação ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
for y, p, cor, nome in [(yv, pv, VERMELHO, "validação"), (yt, pt, AZUL, "teste")]:
    pr, rc, _ = precision_recall_curve(y, p); ax.plot(rc, pr, color=cor, label=f"{nome} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color=cor, ls=":", lw=0.8)
ax.set_xlabel("revocação (= sensibilidade)"); ax.set_ylabel("precisão"); ax.legend(frameon=False); ax.set_title("Precisão x revocação (pontilhado = prevalência)")
salvar(fig, "03_precisao_revocacao.png")

# 4) trade-off do limiar --------------------------------------------------------------------------
ts = np.linspace(0.01, 0.99, 99)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, y, p, nome in [(axs[0], yv, pv, "validação"), (axs[1], yt, pt, "teste")]:
    m = [ponto(y, p, t) for t in ts]
    ax.plot(ts, [q["sensibilidade"] for q in m], color=VERMELHO, label="sensibilidade")
    ax.plot(ts, [q["especificidade"] for q in m], color=AZUL, label="especificidade")
    ax.plot(ts, [q["precisao"] for q in m], color="#2a9d8f", label="precisão")
    ax.plot(ts, [q["acuracia"] for q in m], color=CINZA, ls="--", label="acurácia")
    ax.axhline(1 - y.mean(), color="gray", ls=":", lw=0.9)
    ax.axvline(thr, color="black"); ax.text(thr + 0.01, 0.03, f"limiar {thr:.2f}", fontsize=8)
    ax.set_title(nome); ax.set_xlabel("limiar de decisão")
axs[0].legend(frameon=False, fontsize=8); fig.suptitle("Sensibilidade x especificidade conforme o limiar (pontilhado = acurácia de 'sempre benigno')", y=1.0)
salvar(fig, "04_tradeoff_do_limiar.png")

# 5) matrizes de confusão no teste -----------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, t_ in zip(axs, (0.5, thr)):
    cm = confusion_matrix(yt, (pt >= t_).astype(int), labels=[0, 1]); pc = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pc, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pc[i, j]:.0%})", ha="center", va="center", color="white" if pc[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["previsto\nbenigno", "previsto\nmelanoma"]); ax.set_yticks([0, 1], ["real\nbenigno", "real\nmelanoma"]); ax.set_title(f"teste, limiar {t_:.2f}")
salvar(fig, "05_matrizes_de_confusao_teste.png")

# 6) distribuição dos scores ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, y, p, nome in [(axs[0], yv, pv, "validação"), (axs[1], yt, pt, "teste")]:
    b = np.linspace(0, 1, 26)
    ax.hist(p[y == 0], bins=b, alpha=0.65, color=AZUL, label="benigno (nevo)", density=True)
    ax.hist(p[y == 1], bins=b, alpha=0.65, color=VERMELHO, label="melanoma", density=True)
    ax.axvline(thr, color="black"); ax.set_title(nome); ax.set_xlabel("probabilidade de melanoma"); ax.set_yscale("log")
axs[0].set_ylabel("densidade (escala log)"); axs[0].legend(frameon=False)
salvar(fig, "06_distribuicao_dos_scores.png")

# 7) calibração (teste) -------------------------------------------------------------------------------------
q = np.quantile(pt, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(pt, q[1:-1]), 0, 9)
mp = [pt[idx == k].mean() for k in range(10) if (idx == k).any()]; fr = [yt[idx == k].mean() for k in range(10) if (idx == k).any()]
fig, ax = plt.subplots(figsize=(5.4, 5))
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.plot(mp, fr, marker="o", color=AZUL)
ax.set_xlabel("probabilidade prevista (média do grupo)"); ax.set_ylabel("fração real de melanomas"); ax.set_title("Calibração no teste (10 grupos de mesmo tamanho)")
salvar(fig, "07_calibracao_teste.png")

# 8) por fonte / modalidade na validação (só versões que misturam datasets) ------------------------------------------
por_fonte = {}
ds = globals().get("datasets")
dfv = ds["val"].df.copy() if ds is not None and "val" in ds else None
if dfv is not None and len(dfv) == len(yv) and any(c in dfv for c in ("dataset", "image_type")):
    dfv["p"], dfv["y"] = pv, yv
    cols = [c for c in ("dataset", "image_type") if c in dfv]
    rotulos, aucs, ns = [], [], []
    for chave, g in dfv.groupby(cols):
        if g["y"].nunique() == 2:
            chave = chave if isinstance(chave, tuple) else (chave,)
            rotulos.append(" / ".join(str(c) for c in chave)); aucs.append(roc_auc_score(g["y"], g["p"])); ns.append((int(g["y"].sum()), int((g["y"] == 0).sum())))
    por_fonte = {r: {"auc": float(a), "n_mel": n[0], "n_nv": n[1]} for r, a, n in zip(rotulos, aucs, ns)}
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rotulos) + 1.5))
    ax.barh(rotulos, aucs, color=AZUL)
    for i, (a, n) in enumerate(zip(aucs, ns)):
        ax.text(a + 0.005, i, f"{a:.3f}  ({n[0]} mel / {n[1]} nv)", va="center", fontsize=8)
    ax.set_xlim(0.5, 1.05); ax.set_xlabel("AUC na validação"); ax.set_title("AUC de validação por fonte e modalidade")
    salvar(fig, "08_auc_validacao_por_fonte.png")

# resumo em JSON + zip para baixar -------------------------------------------------------------------------------------
resumo = {"versao": NOME_VERSAO, "auc_validacao": float(auc_v), "auc_teste": float(auc_t), "auc_teste_ic95_bootstrap_de_imagem": ic_t,
          "limiar_escolhido": thr, "teste_no_limiar": ponto(yt, pt, thr), "teste_em_0_50": ponto(yt, pt, 0.5),
          "validacao_no_limiar": ponto(yv, pv, thr), "prevalencia_melanoma_teste": float(yt.mean()),
          "acuracia_sempre_benigno_teste": float(1 - yt.mean()), "auc_validacao_por_fonte": por_fonte,
          "limiares_por_modalidade": globals().get("thr_mod"),
          "historico": [{"epoca": i + 1, "estagio": 1 if i < len(hist1) else 2,
                         **{f"{s}_{k}": float(h[s][k]) for s in ("train", "val") for k in ("loss", "auc", "recall_mel")}}
                        for i, h in enumerate(hist1 + hist2)]}
(OUT / f"resumo_{NOME_VERSAO}.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
zip_path = shutil.make_archive(f"{SAIDA_BASE}/graficos_{NOME_VERSAO}", "zip", OUT)
print(f"\n{NOME_VERSAO}: AUC val {auc_v:.3f} | AUC teste {auc_t:.3f} [{ic_t[0]:.3f}-{ic_t[1]:.3f}] | limiar {thr:.2f} | "
      f"teste: sens {resumo['teste_no_limiar']['sensibilidade']:.3f}, espec {resumo['teste_no_limiar']['especificidade']:.3f}, "
      f"acurácia {resumo['teste_no_limiar']['acuracia']:.3f}")
print("Arquivos:", sorted(p.name for p in OUT.iterdir()))
print("ZIP para baixar:", zip_path)
