"""
Sprint 2 (continuação) — combina HAM10000 + ISIC2020 (mel/nv) num único manifesto de treino,
mantendo o split de VALIDAÇÃO e TESTE do HAM10000 intocado (ver decisão registrada na sessão:
"Opção A" — isola o efeito de mais dado de treino, sem misturar com mudança no conjunto de teste).

Pré-requisito: scripts/prepare_isic2020.py já rodado (extrai e dedupe o ISIC2020 contra o HAM10000).

O que faz:
1. Redimensiona as imagens do ISIC2020 (mel/nv, já dedupe) para 224x224, mesmo padrão do HAM10000.
2. Faz um split por LESÃO só dentro do ISIC2020 (a maioria é 1 imagem por lesão, mas 5 lesões têm
   2+), reservando uma fatia pequena para validação e o resto para treino. Nada de ISIC2020 vai
   para teste.
3. Junta com as linhas ORIGINAIS do HAM10000 (descarta as cópias fixas de augmentation da Sprint 1
   — o notebook de treino já usa augmentation aleatório a cada época via `strong_aug=True`,
   então as cópias fixas não são mais necessárias e só inflariam o manifesto à toa).
4. Valida: nenhuma lesão do HAM10000 nem do ISIC2020 aparece em mais de um split; test/val do
   HAM10000 têm exatamente as mesmas imagens de antes.

Uso:
    python scripts/build_combined_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
HAM_SPLIT_INDEX = BASE_DIR / "data" / "processed" / "split_index.csv"
ISIC_METADATA = BASE_DIR / "data" / "raw" / "isic2020" / "isic2020_mel_nv_metadata.csv"
ISIC_RAW_DIR = BASE_DIR / "data" / "raw" / "isic2020" / "images"
RESIZED_DIR = BASE_DIR / "data" / "processed" / "images_224"
OUT_PATH = BASE_DIR / "data" / "processed" / "split_index_combined.csv"

IMG_SIZE = (224, 224)
VAL_FRAC_ISIC = 0.10
SEED = 42


def resize_isic_images(df: pd.DataFrame) -> None:
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    faltando = [row for row in df.itertuples() if not (RESIZED_DIR / f"{row.image_name}.jpg").exists()]
    if not faltando:
        print(f"      {len(df)} imagens já redimensionadas, pulando.")
        return
    print(f"      Redimensionando {len(faltando)} de {len(df)} imagens do ISIC2020...")
    for row in tqdm(faltando, unit="img"):
        src = ISIC_RAW_DIR / f"{row.image_name}.jpg"
        dst = RESIZED_DIR / f"{row.image_name}.jpg"
        with Image.open(src) as img:
            img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR).save(dst, "JPEG", quality=95)


def split_isic_by_lesion(df: pd.DataFrame, seed: int) -> pd.Series:
    lesion_df = df.drop_duplicates(subset="lesion_id")[["lesion_id", "dx"]].set_index("lesion_id")
    idx_train, idx_val = train_test_split(
        lesion_df.index, test_size=VAL_FRAC_ISIC, stratify=lesion_df["dx"], random_state=seed
    )
    split = pd.Series(index=lesion_df.index, dtype=object)
    split.loc[idx_train] = "train"
    split.loc[idx_val] = "val"
    return df["lesion_id"].map(split)


def validate(combined: pd.DataFrame, ham_original: pd.DataFrame) -> None:
    # 1) nenhuma lesão em mais de um split (checa os dois datasets separadamente - os prefixos
    #    HAM_/IL_ não colidem, mas separar deixa a mensagem de erro mais clara)
    for fonte, sub in [("HAM10000", combined[combined["dataset"] == "ham10000"]),
                       ("ISIC2020", combined[combined["dataset"] == "isic2020"])]:
        vazando = sub.groupby("lesion_id")["split"].nunique()
        vazando = vazando[vazando > 1]
        assert len(vazando) == 0, f"{fonte}: {len(vazando)} lesões vazando entre splits"
    print("      Validação OK: nenhuma lesão (HAM10000 ou ISIC2020) aparece em mais de um split.")

    # 2) val/test do HAM10000 idênticos aos do split original (Opção A: não mexer neles)
    for split in ("val", "test"):
        antes = set(ham_original[ham_original["split"] == split]["image_id"])
        depois = set(combined[(combined["dataset"] == "ham10000") & (combined["split"] == split)]["image_id"])
        assert antes == depois, f"split '{split}' do HAM10000 mudou - isso quebra a Opção A"
    print("      Validação OK: val e test do HAM10000 são exatamente os mesmos de antes.")


def main() -> None:
    if not ISIC_METADATA.exists():
        raise FileNotFoundError(f"{ISIC_METADATA} não encontrado. Rode scripts/prepare_isic2020.py primeiro.")

    print("1/4 - Carregando metadados do ISIC2020 e redimensionando imagens...")
    isic = pd.read_csv(ISIC_METADATA)
    resize_isic_images(isic)

    print("2/4 - Split por lesão do ISIC2020 (só train/val - nunca test)...")
    isic["split"] = split_isic_by_lesion(isic, SEED)
    isic_out = pd.DataFrame({
        "image_id": isic["image_name"],
        "lesion_id": isic["lesion_id"],
        "path": [str((RESIZED_DIR / f"{n}.jpg").relative_to(BASE_DIR)) for n in isic["image_name"]],
        "split": isic["split"],
        "dx": isic["dx"],
        "dx_type": "isic2020_confirmed",
        "is_augmented": False,
        "source_image_id": isic["image_name"],
        "dataset": "isic2020",
    })
    print(f"      {(isic_out['split']=='train').sum()} treino / {(isic_out['split']=='val').sum()} validação "
          f"({isic_out.groupby('split')['dx'].value_counts().to_dict()})")

    print("3/4 - Combinando com as imagens ORIGINAIS do HAM10000 (sem as cópias fixas de augmentation)...")
    ham = pd.read_csv(HAM_SPLIT_INDEX)
    ham_original = ham[~ham["is_augmented"]].copy()
    ham_original["dataset"] = "ham10000"
    cols = ["image_id", "lesion_id", "path", "split", "dx", "dx_type", "is_augmented", "source_image_id", "dataset"]
    combined = pd.concat([ham_original[cols], isic_out[cols]], ignore_index=True)

    print("4/4 - Validando e salvando...")
    validate(combined, ham_original)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nManifesto combinado: {OUT_PATH}")
    print("\nComposição final por split e classe:")
    print(combined.groupby(["split", "dx"]).size().unstack())
    print("\nComposição final por split e fonte:")
    print(combined.groupby(["split", "dataset"]).size().unstack())


if __name__ == "__main__":
    main()
