"""
TCC-03 — Split treino/validação/teste (70/15/15) do HAM10000 (mel vs nv).

Diferença crítica em relação a um split "ingênuo": 1.801 lesões (de 6.017 mel+nv) têm
MAIS DE UMA FOTO. Se o split fosse feito por IMAGEM, a mesma lesão poderia aparecer em
treino e em teste ao mesmo tempo (duas fotos levemente diferentes da mesma pinta) — o
modelo "veria" o dado de teste durante o treino de forma disfarçada. Isso é um erro
metodológico clássico e bem documentado na literatura que usa HAM10000, e é o motivo
pelo qual este split é feito por LESÃO (lesion_id), não por imagem:

    1. Reduz o dataset ao nível de lesão única (1 linha por lesion_id).
    2. Faz o split estratificado por classe (dx: mel/nv) nesse nível.
    3. Propaga o split de volta pra todas as imagens daquela lesão (incluindo as
       aumentadas geradas pela TCC-02, que herdam o split da imagem-fonte).

Este dataset NÃO tem tom de pele — a estratificação aqui é só por classe (mel/nv).
A análise de fairness por tom de pele deste TCC usa o DDI como benchmark de avaliação
separado (ver scripts/prepare_ddi.py), não este split.

Uso:
    python scripts/split_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "preprocessed_manifest.csv"
SPLIT_INDEX_PATH = PROCESSED_DIR / "split_index.csv"

VAL_FRAC = 0.15
TEST_FRAC = 0.15


def split_by_lesion(lesion_df: pd.DataFrame, seed: int) -> pd.Series:
    """70/15/15 estratificado por classe, no nível de lesão (1 linha = 1 lesion_id)."""
    idx_train, idx_temp = train_test_split(
        lesion_df.index, test_size=(VAL_FRAC + TEST_FRAC), stratify=lesion_df["dx"], random_state=seed
    )
    temp_classes = lesion_df.loc[idx_temp, "dx"]
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC), stratify=temp_classes, random_state=seed
    )
    split = pd.Series(index=lesion_df.index, dtype=object)
    split.loc[idx_train] = "train"
    split.loc[idx_val] = "val"
    split.loc[idx_test] = "test"
    return split


def validate_no_lesion_leakage(manifest: pd.DataFrame) -> None:
    """Confere programaticamente que nenhuma lesão aparece em mais de um split."""
    lesion_splits = manifest.groupby("lesion_id")["split"].nunique()
    vazando = lesion_splits[lesion_splits > 1]
    if len(vazando):
        print("ERRO DE VALIDAÇÃO — lesões aparecendo em mais de um split (vazamento de dados):")
        print(vazando)
        raise AssertionError(f"{len(vazando)} lesões vazando entre splits — split inválido.")
    print(f"Validação OK: nenhuma das {manifest['lesion_id'].nunique()} lesões aparece em mais de um split.")


def print_relatorio(manifest: pd.DataFrame) -> None:
    print("\nTamanho de cada split (imagens, originais + aumentadas):")
    print(manifest["split"].value_counts())
    print(manifest["split"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    print("\nPercentual de cada classe DENTRO de cada split:")
    print(pd.crosstab(manifest["split"], manifest["dx"], normalize="index").mul(100).round(1))

    originais = manifest[~manifest["is_augmented"]]
    print("\nSó imagens ORIGINAIS por split (sem contar augmentation):")
    print(pd.crosstab(originais["split"], originais["dx"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} não encontrado. Rode scripts/preprocess.py (TCC-02) primeiro.")

    manifest = pd.read_csv(MANIFEST_PATH)
    originais = manifest[~manifest["is_augmented"]].copy()

    # reduz ao nivel de lesao (1 lesao pode ter N fotos, mas so 1 classe - ja validado na TCC-02/EDA)
    lesion_df = originais.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]].set_index("lesion_id")
    print(f"Lesões únicas (mel+nv): {len(lesion_df)}  |  Imagens originais: {len(originais)}")

    lesion_split = split_by_lesion(lesion_df, args.seed)

    originais["split"] = originais["lesion_id"].map(lesion_split)

    # propaga o split para as copias aumentadas via a lesao da imagem-fonte
    split_por_imagem_fonte = originais.set_index("image_id")["split"]
    lesion_por_imagem_fonte = originais.set_index("image_id")["lesion_id"]
    aumentadas = manifest[manifest["is_augmented"]].copy()
    aumentadas["split"] = aumentadas["source_image_id"].map(split_por_imagem_fonte)
    aumentadas["lesion_id"] = aumentadas["source_image_id"].map(lesion_por_imagem_fonte)

    completo = pd.concat([originais, aumentadas], ignore_index=True)

    print("\n--- Validação ---")
    validate_no_lesion_leakage(completo)
    print_relatorio(completo)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    completo[[
        "image_id", "lesion_id", "path", "split", "dx", "dx_type", "is_augmented", "source_image_id",
    ]].to_csv(SPLIT_INDEX_PATH, index=False)

    print(f"\nÍndice de split salvo em: {SPLIT_INDEX_PATH}")
    print(f"Total no índice (originais + aumentadas): {len(completo)}")
    print("\nLEMBRETE: use augmentation (is_augmented=True) só nas linhas com split=='train'.")
    print("Fairness por tom de pele: avaliar o modelo treinado aqui sobre o DDI (script separado), "
          "não sobre este split de teste.")


if __name__ == "__main__":
    main()
