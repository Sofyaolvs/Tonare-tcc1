"""
v6 — mesmos dados da v5 (HAM10000 + MILK10k + BCN20000), mas com as FOTOS CLÍNICAS do MILK10k
repetidas mais vezes no treino.

Motivo: a v4 (clínicas = ~10% do treino) foi a melhor em foto clínica (AUC 0,79 no MILK10k reservado);
na v5 o BCN20000 diluiu as clínicas para ~4% e a AUC clínica voltou para 0,72, enquanto a AUC em
dermatoscopia real (MSK 2020) subiu de 0,67 para 0,76. Hipótese: sobreamostrar as clínicas mantém
os dois ganhos. É uma hipótese, não está provada.

Desenho (só UMA mudança em relação à v5, para a comparação ser limpa):
  - split_index_v5.csv inalterado: val, test, test_milk e as lesões de cada split idênticos.
  - Linhas de treino que são foto clínica do MILK10k aparecem CLINICAL_COPIES vezes. O augmentation
    do notebook é aleatório a cada época, então cópias repetidas não são vistas iguais.
  - Nada de val/teste é repetido.

Uso:
    python scripts/build_v6_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
V5 = BASE_DIR / "data" / "processed" / "split_index_v5.csv"
OUT = BASE_DIR / "data" / "processed" / "split_index_v6.csv"
CLINICAL_COPIES = 4  # cada foto clínica do treino aparece 4x (3 cópias extras)


def main() -> None:
    v5 = pd.read_csv(V5)
    is_clin_train = (v5["dataset"] == "milk10k") & (v5["image_type"] == "clinical") & (v5["split"] == "train")
    extra = pd.concat([v5[is_clin_train]] * (CLINICAL_COPIES - 1), ignore_index=True)
    v6 = pd.concat([v5, extra], ignore_index=True)

    for s in ("val", "test", "test_milk"):
        a = v5[v5["split"] == s].reset_index(drop=True)
        b = v6[v6["split"] == s].reset_index(drop=True)
        assert a.equals(b), f"split '{s}' mudou"
    assert (v6.groupby("lesion_id")["split"].nunique() > 1).sum() == 0, "lesão em mais de um split"
    print("Validação OK: val/test/test_milk idênticos aos da v5; nenhuma lesão em dois splits.")
    v6.to_csv(OUT, index=False)

    tr = v6[v6["split"] == "train"]
    print(f"\nManifesto v6: {OUT}")
    print(f"Treino: {len(tr)} imagens (v5: {int((v5['split'] == 'train').sum())})")
    print("Fração por fonte/modalidade no treino:")
    print((tr.groupby(["dataset", "image_type"]).size() / len(tr)).round(3).to_string())
    print("\nTreino por classe:", tr["dx"].value_counts().to_dict())


if __name__ == "__main__":
    main()
