"""
Gera, LOCALMENTE (sem Kaggle), os gráficos de treino/validação/teste de cada versão do modelo (v2 a v7),
com a mesma célula que roda nos notebooks do Kaggle (scripts/celula_graficos_treino.py).

O que vem de onde:
  - probabilidades de validação e teste: recalculadas aqui a partir do checkpoint e do manifesto da versão
    (reproduzem exatamente a val_auc salva no checkpoint; o script confere);
  - curvas de treino (loss/AUC por época): extraídas dos LOGS que estão dentro dos notebooks do Kaggle
    (linhas "[Estágio N] época ...") — a v3 não tem saídas salvas, então fica sem curva.

Saídas:
    docs/figures/por_versao/graficos_<versao>/*.png e resumo_<versao>.json
    docs/data/treino_historico_por_versao.csv
    docs/data/resumo_por_versao_notebook.csv

Uso:
    python scripts/gerar_graficos_por_versao.py [pasta_dos_notebooks]   # padrão: notebooks/kaggle_executados
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
DOCS = BASE / "docs"
NB_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "notebooks" / "kaggle_executados"
CELULA = (BASE / "scripts" / "celula_graficos_treino.py").read_text(encoding="utf-8")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# versão -> (manifesto usado na validação/teste, notebook executado no Kaggle com o log). Os notebooks exportados do
# Kaggle tinham nomes deslocados em relação às versões; as cópias em notebooks/kaggle_executados/ já foram renomeadas
# conforme a numeração do projeto (mapeamento conferido pelo tamanho do treino, AUC e limiar de cada log).
VERSOES = {
    "v2": ("split_index.csv", "tonare_v2_executado.ipynb"),
    "v3": ("split_index_combined.csv", "tonare_v3_executado.ipynb"),
    "v4": ("split_index_v4.csv", "tonare_v4_executado.ipynb"),
    "v5": ("split_index_v5.csv", "tonare_v5_executado.ipynb"),
    "v6": ("split_index_v6.csv", "tonare_v6_executado.ipynb"),
    "v7": ("split_index_v7.csv", "tonare_v7_executado.ipynb"),
}
LINHA = re.compile(r"\[Estágio (\d)\] época (\d+)/(\d+) \| treino: loss=([\d.]+) auc=([\d.]+) \| "
                   r"val: loss=([\d.]+) auc=([\d.]+) recall_mel@0\.5=([\d.]+)")


def historico(nb_path: Path):
    if not nb_path.exists():
        return [], []
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    h = {1: [], 2: []}
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                for m in LINHA.finditer("".join(out.get("text", []))):
                    est, ep, _, tl, ta, vl, va, vr = m.groups()
                    h[int(est)].append({"epoch": int(ep), "train": {"loss": float(tl), "auc": float(ta), "recall_mel": float("nan")},
                                        "val": {"loss": float(vl), "auc": float(va), "recall_mel": float(vr)}})
    # um notebook pode ter sido rodado mais de uma vez: fica a última execução de cada estágio
    for est in (1, 2):
        lst, ultimo = h[est], []
        for e in lst:
            if e["epoch"] == 1:
                ultimo = []
            ultimo.append(e)
        h[est] = ultimo
    return h[1], h[2]


def pontuar(model, df) -> np.ndarray:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    paths = [BASE / p.replace("\\", "/") for p in df["path"]]
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            out += torch.softmax(model(torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 64]])), 1)[:, 1].tolist()
    return np.array(out)


def main() -> None:
    hist_rows, resumo_rows = [], []
    for v, (manifesto, nb_nome) in VERSOES.items():
        ck = torch.load(PROC / f"model_resnet50_baseline_{v}.pt", map_location="cpu", weights_only=True)
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        man = pd.read_csv(PROC / manifesto)
        man = man[~man["is_augmented"]]
        val_df = man[man["split"] == "val"].reset_index(drop=True)
        test_df = man[man["split"] == "test"].reset_index(drop=True)
        cache = PROC / f"valtest_scores_{v}.csv"
        if cache.exists():
            sc = pd.read_csv(cache)
            pv, pt = sc[sc["split"] == "val"]["p"].to_numpy(), sc[sc["split"] == "test"]["p"].to_numpy()
        else:
            print(f"[{v}] pontuando validação ({len(val_df)}) e teste ({len(test_df)}) na CPU...", flush=True)
            pv, pt = pontuar(model, val_df), pontuar(model, test_df)
            pd.concat([pd.DataFrame({"split": "val", "p": pv}), pd.DataFrame({"split": "test", "p": pt})]).to_csv(cache, index=False)
        yv, yt = (val_df["dx"] == "mel").astype(int).to_numpy(), (test_df["dx"] == "mel").astype(int).to_numpy()
        auc_val = roc_auc_score(yv, pv)
        ok = abs(auc_val - float(ck["val_auc"])) < 2e-3
        print(f"[{v}] val AUC recalculada {auc_val:.4f} | salva no checkpoint {float(ck['val_auc']):.4f} | {'confere' if ok else 'DIVERGE'}", flush=True)
        assert ok, f"{v}: a validação recalculada não bate com a do checkpoint (manifesto errado?)"

        h1, h2 = historico(NB_DIR / nb_nome)
        print(f"[{v}] log do notebook '{nb_nome}': {len(h1)} épocas no estágio 1, {len(h2)} no estágio 2", flush=True)
        for est, h in ((1, h1), (2, h2)):
            for e in h:
                hist_rows.append({"versao": v, "estagio": est, "epoca": e["epoch"], "train_loss": e["train"]["loss"], "train_auc": e["train"]["auc"],
                                  "val_loss": e["val"]["loss"], "val_auc": e["val"]["auc"], "val_recall_mel_0_5": e["val"]["recall_mel"]})

        g = {"val_labels": yv, "val_probs": pv, "test_labels": yt, "test_probs": pt, "best_threshold": float(ck["threshold"]),
             "history_stage1": h1, "history_stage2": h2, "datasets": {"val": types.SimpleNamespace(df=val_df)},
             "checkpoint_path": f"model_resnet50_baseline_{v}.pt"}
        if ck.get("thresholds_by_modality"):
            g["thr_mod"] = ck["thresholds_by_modality"]
        destino = DOCS / "figures" / "por_versao"
        destino.mkdir(parents=True, exist_ok=True)
        codigo = (CELULA.replace('NOME_VERSAO = None ', f'NOME_VERSAO = "{v}" ')
                  .replace('SAIDA_BASE = "/kaggle/working"', f'SAIDA_BASE = "{destino.as_posix()}"'))
        exec(compile(codigo, f"celula_{v}", "exec"), g)
        (destino / f"graficos_{v}.zip").unlink(missing_ok=True)  # o zip é só para download no Kaggle; aqui seria duplicata
        resumo = json.loads((destino / f"graficos_{v}" / f"resumo_{v}.json").read_text(encoding="utf-8"))
        resumo_rows.append({"versao": v, "n_val": len(yv), "n_teste": len(yt), "n_mel_teste": int(yt.sum()), "auc_validacao": resumo["auc_validacao"],
                            "auc_teste": resumo["auc_teste"], "limiar": resumo["limiar_escolhido"],
                            "sens_teste": resumo["teste_no_limiar"]["sensibilidade"], "espec_teste": resumo["teste_no_limiar"]["especificidade"],
                            "acuracia_teste": resumo["teste_no_limiar"]["acuracia"], "precisao_teste": resumo["teste_no_limiar"]["precisao"],
                            "melanomas_perdidos_teste": resumo["teste_no_limiar"]["fn"], "nevos_marcados_teste": resumo["teste_no_limiar"]["fp"],
                            "epocas_estagio1": len(h1), "epocas_estagio2": len(h2)})
        (DOCS / "data").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(hist_rows).to_csv(DOCS / "data" / "treino_historico_por_versao.csv", index=False)
        pd.DataFrame(resumo_rows).to_csv(DOCS / "data" / "resumo_por_versao_notebook.csv", index=False)
    print("\nPronto. Resumo:")
    print(pd.DataFrame(resumo_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
