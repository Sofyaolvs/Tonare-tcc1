"""
Limiar de decisão POR MODALIDADE (dermatoscopia x foto clínica), sem retreinar.

Os checkpoints salvam um limiar único, escolhido numa validação que mistura modalidades; em foto
clínica isso dá sensibilidade alta e especificidade de 0,1-0,3. Aqui os limiares são escolhidos
SÓ na validação de cada modelo, separados por modalidade, e depois aplicados aos testes externos
(que NUNCA entram na escolha). Duas políticas:
  A - maior limiar com sensibilidade >= 0,90 na validação da modalidade (a regra do projeto)
  B - Youden (sens + espec - 1 máxima), um ponto equilibrado
Validação usada: v2 e v4 -> val do manifesto v4 (HAM10000 + MILK10k); v5 e v6 -> val do manifesto v5/v6 (+ BCN20000; val idêntica). ATENÇÃO: a validação CLÍNICA tem só 44 melanomas / 74 nevos, então o limiar clínico é ruidoso.

Uso:
    python scripts/threshold_by_modality.py v2 v4 v5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
PROC = BASE_DIR / "data" / "processed"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TARGET_SENS = 0.90
N_BOOT, SEED = 1000, 42
GRID = np.round(np.arange(0.01, 0.99, 0.01), 2)


def load_model(v: str):
    ck = torch.load(PROC / f"model_resnet50_baseline_{v}.pt", map_location="cpu", weights_only=True)
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, float(ck["threshold"])


@torch.no_grad()
def score_paths(model, paths) -> np.ndarray:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    out = []
    for i in range(0, len(paths), 64):
        batch = torch.stack([tf(Image.open(BASE_DIR / p.replace("\\", "/")).convert("RGB")) for p in paths[i:i + 64]])
        out.extend(torch.softmax(model(batch), dim=1)[:, 1].tolist())
    return np.array(out)


def sens_spec(y, s, t):
    pred = s >= t
    return (float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
            float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan"))


def pick(y, s) -> dict:
    curve = [(t, *sens_spec(y, s, t)) for t in GRID]
    ok = [t for t, se, _ in curve if se >= TARGET_SENS]
    a = max(ok) if ok else max(curve, key=lambda r: (r[1], r[0]))[0]
    b = max(curve, key=lambda r: (r[1] + r[2], r[0]))[0]
    return {"A (sens>=0,90)": float(a), "B (Youden)": float(b)}


def boot_ci(y, s, t, groups, rng):
    idx_by = list(pd.Series(range(len(y))).groupby(np.asarray(groups)).indices.values())
    se_l, sp_l = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by[i] for i in rng.integers(0, len(idx_by), len(idx_by))])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        se, sp = sens_spec(yb, s[idx], t)
        se_l.append(se)
        sp_l.append(sp)
    q = lambda v: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
    return q(se_l), q(sp_l)


def load_tests(v: str, model) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """(nome, modalidade, y, score, grupo) de cada teste externo."""
    tests = []
    ham = pd.read_csv(PROC / "split_index_v4.csv")
    ham = ham[(ham["split"] == "test") & (ham["dataset"] == "ham10000")].reset_index(drop=True)
    tests.append(("HAM10000 teste", "dermatoscopia", (ham["dx"] == "mel").astype(int).to_numpy(),
                  score_paths(model, list(ham["path"])), ham["lesion_id"].to_numpy()))
    mk = pd.read_csv(PROC / "milk10k" / f"milk10k_predictions_{v}_heldout.csv")
    for mod in ("dermatoscopia", "foto clínica"):
        g = mk[mk["modalidade"] == mod]
        tests.append((f"MILK10k reservado", mod, g["y"].to_numpy(), g["score_mel"].to_numpy(), g["lesion_id"].to_numpy()))
    pad = pd.read_csv(PROC / "padufes20" / f"padufes20_predictions_{v}.csv")
    tests.append(("PAD-UFES-20", "foto clínica", pad["y"].to_numpy(), pad["score_mel"].to_numpy(), pad["lesion_id"].to_numpy()))
    hb = pd.read_csv(PROC / "external" / f"hiba251_{v}.csv")
    for mod in ("dermatoscopia", "foto clínica"):
        g = hb[hb["modalidade"] == mod]
        tests.append(("HIBA", mod, g["y"].to_numpy(), g["score_mel"].to_numpy(), g["grp"].to_numpy()))
    ms = pd.read_csv(PROC / "external" / f"consec_mel2020_216_{v}.csv")
    tests.append(("MSK 2020 consecutivas", "dermatoscopia", ms["y"].to_numpy(), ms["score_mel"].to_numpy(), ms["grp"].to_numpy()))
    dd = pd.read_csv(PROC / "ddi" / ("ddi_predictions.csv" if v == "v2" else f"ddi_predictions_{v}.csv"))
    tests.append(("DDI (maligno vs benigno)", "foto clínica", dd["y"].to_numpy(), dd["score_mel"].to_numpy(), np.arange(len(dd))))
    fzf = PROC / "fitzpatrick17k" / f"fitz17k_{v}.csv"
    if fzf.exists():  # foto clínica de atlas, só tons V-VI (na v7 são as únicas nunca vistas); melanoma vs nevo
        fz = pd.read_csv(fzf)
        fz = fz[(fz["grupo"] == "V-VI") & (fz["label"] != "lentigo maligna")]
        tests.append(("Fitzpatrick17k V-VI", "foto clínica", fz["y"].to_numpy(), fz["score_mel"].to_numpy(), np.arange(len(fz))))
    return tests


def main(versions: list[str]) -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for v in versions:
        model, thr_global = load_model(v)
        man = pd.read_csv(PROC / {"v5": "split_index_v5.csv", "v6": "split_index_v6.csv", "v7": "split_index_v7.csv"}.get(v, "split_index_v4.csv"))
        val = man[man["split"] == "val"].reset_index(drop=True)
        print(f"\n##### {v}: limiar único salvo = {thr_global:.2f} | pontuando a validação ({len(val)} imgs)...", flush=True)
        val["score"] = score_paths(model, list(val["path"]))
        val["y"] = (val["dx"] == "mel").astype(int)
        thr = {}
        for mod_key, mod in (("dermoscopic", "dermatoscopia"), ("clinical", "foto clínica")):
            g = val[val["image_type"] == mod_key]
            thr[mod] = pick(g["y"].to_numpy(), g["score"].to_numpy())
            print(f"  validação {mod}: {int(g['y'].sum())} mel / {int((g['y'] == 0).sum())} nv -> limiares {thr[mod]}")
        for nome, mod, y, s, grp in load_tests(v, model):
            for pol_name, t in [("único (atual)", thr_global), *[(k, thr[mod][k]) for k in thr[mod]]]:
                se, sp = sens_spec(y, s, t)
                (se_lo, se_hi), (sp_lo, sp_hi) = boot_ci(y, s, t, grp, rng)
                rows.append({"modelo": v, "teste": nome, "modalidade": mod, "politica": pol_name, "limiar": t,
                             "n_mel": int(y.sum()), "n_nv": int((y == 0).sum()),
                             "sens": se, "sens_lo": se_lo, "sens_hi": se_hi, "espec": sp, "espec_lo": sp_lo, "espec_hi": sp_hi})
    res = pd.DataFrame(rows)
    out = PROC / "threshold_by_modality.csv"
    res.to_csv(out, index=False)
    f = lambda v, lo, hi: f"{v:.2f} [{lo:.2f}-{hi:.2f}]"
    for v in versions:
        print(f"\n=== {v} ===")
        print(f"{'teste':<26}{'modalidade':<15}{'política':<16}{'limiar':>7}  {'sensibilidade [IC95%]':<22}{'especificidade [IC95%]':<22}")
        for r in res[res["modelo"] == v].itertuples():
            print(f"{r.teste:<26}{r.modalidade:<15}{r.politica:<16}{r.limiar:>7.2f}  {f(r.sens, r.sens_lo, r.sens_hi):<22}{f(r.espec, r.espec_lo, r.espec_hi):<22}")
    print(f"\nResultados: {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["v2", "v4", "v5"])
