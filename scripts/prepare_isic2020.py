"""
Sprint 2b — Incorporação do ISIC 2020 ao treino mel vs nv (ampliação de dados).

Motivo: o HAM10000 tem só 428 lesões distintas de melanoma no treino, o que limitou o
recall do classificador (ver histórico no notebook 02_train_baseline). O ISIC 2020 tem
584 melanomas e 5.193 nevos CONFIRMADOS (coluna `diagnosis`), fora das 27.124 imagens
"unknown" (benignas mas sem subtipo confirmado — deixadas de fora para não misturar
outro tipo de lesão benigna na classe `nv`).

Extrai só as imagens necessárias do zip de 23GB (mais rápido que descompactar tudo),
detecta duplicatas EXATAS contra o HAM10000 (os dois datasets compartilham a fonte
Medical University of Vienna, então há risco real de sobreposição) e gera um manifesto
no mesmo formato do HAM10000 (lesion_id, dx, path, etc.) para ser combinado depois.

Uso:
    python scripts/prepare_isic2020.py
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = Path("C:/Users/sofya/Downloads/ISIC_2020_Training_JPEG.zip")
GT_PATH = Path("C:/Users/sofya/Downloads/ISIC_2020_Training_GroundTruth_v2.csv")

RAW_DIR = BASE_DIR / "data" / "raw" / "isic2020"
IMAGES_DIR = RAW_DIR / "images"
HAM_IMAGES_DIR = BASE_DIR / "data" / "raw" / "ham10000" / "images"

OUT_METADATA = RAW_DIR / "isic2020_mel_nv_metadata.csv"
DX_MAP = {"melanoma": "mel", "nevus": "nv"}


def extract_needed_images(image_names: list[str]) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    faltando = [n for n in image_names if not (IMAGES_DIR / f"{n}.jpg").exists()]
    if not faltando:
        print(f"      {len(image_names)} imagens já extraídas, pulando.")
        return
    print(f"      Extraindo {len(faltando)} de {len(image_names)} imagens (zip de 23GB, pode levar alguns minutos)...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        for i, name in enumerate(faltando, 1):
            with z.open(f"train/{name}.jpg") as src, open(IMAGES_DIR / f"{name}.jpg", "wb") as dst:
                dst.write(src.read())
            if i % 500 == 0:
                print(f"        {i}/{len(faltando)}")


def md5_of_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def find_duplicates_vs_ham10000(image_names: list[str]) -> set[str]:
    print("      Calculando hashes do HAM10000 (referência)...")
    ham_hashes = {md5_of_file(p): p.name for p in HAM_IMAGES_DIR.glob("*.jpg")}
    print(f"      {len(ham_hashes)} imagens do HAM10000 indexadas.")

    duplicatas = set()
    for name in image_names:
        h = md5_of_file(IMAGES_DIR / f"{name}.jpg")
        if h in ham_hashes:
            duplicatas.add(name)
    return duplicatas


def main() -> None:
    if not ZIP_PATH.exists() or not GT_PATH.exists():
        raise FileNotFoundError(f"Esperado {ZIP_PATH} e {GT_PATH}")

    print("1/4 - Carregando metadados e filtrando melanoma/nevo confirmados...")
    gt = pd.read_csv(GT_PATH)
    df = gt[gt["diagnosis"].isin(DX_MAP)].copy()
    df["dx"] = df["diagnosis"].map(DX_MAP)
    print(f"      {len(df)} imagens ({df['dx'].value_counts().to_dict()}), "
          f"{df['lesion_id'].nunique()} lesões, {df['patient_id'].nunique()} pacientes")

    print("2/4 - Extraindo imagens necessárias do zip...")
    extract_needed_images(df["image_name"].tolist())

    print("3/4 - Checando duplicatas EXATAS contra o HAM10000 (fonte compartilhada: Univ. de Viena)...")
    duplicatas = find_duplicates_vs_ham10000(df["image_name"].tolist())
    print(f"      {len(duplicatas)} duplicatas exatas encontradas e removidas.")
    if duplicatas:
        print(f"      Exemplos: {list(duplicatas)[:5]}")
    df = df[~df["image_name"].isin(duplicatas)].copy()

    print("4/4 - Salvando metadados...")
    df["path"] = df["image_name"].apply(lambda n: str((IMAGES_DIR / f"{n}.jpg").relative_to(BASE_DIR)))
    df["dataset"] = "isic2020"
    out = df[["image_name", "lesion_id", "patient_id", "dx", "diagnosis", "path", "dataset"]]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_METADATA, index=False)

    print(f"\nConcluído: {len(out)} imagens novas do ISIC 2020 ({out['dx'].value_counts().to_dict()})")
    print(f"Metadados: {OUT_METADATA}")
    print(f"Imagens: {IMAGES_DIR}")
    print(f"\nSomado ao HAM10000 (771 mel / 4584 nv só no treino): "
          f"mel {771 + int((out['dx']=='mel').sum())}, nv {4584 + int((out['dx']=='nv').sum())} "
          f"(estimativa - split real ainda precisa ser refeito combinando os dois).")


if __name__ == "__main__":
    main()
