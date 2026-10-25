"""
Sorteia os casos de exemplo da demonstração (scripts/frontend/demo_app.py).

Regra (declarada ANTES de olhar a saída do modelo, para não "escolher a dedo"):
  - só imagens que NUNCA entraram em nenhum treino: teste do HAM10000 e PAD-UFES-20;
  - sorteio simples com semente fixa, balanceado por classe: 2 melanomas + 2 nevos em dermatoscopia (HAM10000)
    e 1 melanoma + 1 nevo em fotografia clínica (PAD-UFES-20, fotos de celular do Brasil);
  - as imagens são gravadas no mesmo formato do treino (224x224 bilinear + JPEG q95).
A semente 42 é a usada na demonstração. O arquivo data/demo_examples/exemplos.csv guarda a origem e o rótulo
REAL de cada caso (a interface não mostra o rótulo; é para quem apresenta saber).

Uso:
    python scripts/select_demo_examples.py [semente]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
OUT = BASE / "data" / "demo_examples"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
CKPT_VERSION = "v7"


def salvar_224(origem: Path, destino: Path) -> None:
    img = Image.open(origem).convert("RGB").resize((224, 224), Image.BILINEAR)
    img.save(destino, "JPEG", quality=95)


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- dermatoscopia: teste do HAM10000 (a ordem das previsões é a do manifesto, sem imagens aumentadas) ---
    man = pd.read_csv(PROC / f"split_index_{CKPT_VERSION}.csv")
    man = man[(man["split"] == "test") & (man["dataset"] == "ham10000") & (~man["is_augmented"])].reset_index(drop=True)
    sc = pd.read_csv(PROC / f"valtest_scores_{CKPT_VERSION}.csv")
    sc = sc[sc["split"] == "test"].reset_index(drop=True)
    assert len(man) == len(sc), "manifesto e previsões do teste não alinham"
    ham = man.assign(score=sc["p"])
    # --- fotografia clínica: PAD-UFES-20 (nunca usado no treino) ---
    pad = pd.read_csv(PROC / "padufes20" / f"padufes20_predictions_{CKPT_VERSION}.csv")

    escolhidos = []
    for dx, n in (("mel", 2), ("nv", 2)):
        pool = ham[ham["dx"] == dx]["image_id"].tolist()
        for iid in rng.sample(sorted(pool), n):
            escolhidos.append(("HAM10000 (teste)", "Dermatoscopia", dx, PROC / "images_224" / f"{iid}.jpg", iid,
                               float(ham.loc[ham["image_id"] == iid, "score"].iloc[0])))
    for y, dx in ((1, "mel"), (0, "nv")):
        pool = pad[pad["y"] == y]["img_id"].tolist()
        iid = rng.choice(sorted(pool))
        escolhidos.append(("PAD-UFES-20", "Fotografia clínica", dx, BASE / "data" / "raw" / "padufes20" / "images" / iid, iid,
                           float(pad.loc[pad["img_id"] == iid, "score_mel"].iloc[0])))

    # embaralha a ordem de exibição (com a mesma semente) para não ficar "primeiro melanomas, depois nevos"
    rng.shuffle(escolhidos)
    for antigo in list(OUT.glob("caso_*.jpg")) + list(OUT.glob("exemplo_*.jpg")):
        antigo.unlink()
    linhas = []
    for i, (fonte, modalidade, dx, origem, iid, score) in enumerate(escolhidos, 1):
        nome = f"exemplo_{i:02d}.jpg"
        salvar_224(origem, OUT / nome)
        linhas.append({"arquivo": nome, "modalidade": modalidade, "fonte": fonte, "id_origem": iid, "rotulo_real": dx,
                       "pontuacao_v7_na_avaliacao": round(score, 4)})
    pd.DataFrame(linhas).to_csv(OUT / "exemplos.csv", index=False, encoding="utf-8")
    print(f"semente {SEED}: {len(linhas)} exemplos gravados em {OUT}")
    print(pd.DataFrame(linhas).to_string(index=False))


if __name__ == "__main__":
    main()
