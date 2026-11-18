"""
v5 — soma o BCN20000 (dermatoscopia, Hospital Clínic de Barcelona, mel/nv) ao manifesto da v4.

Motivo: o BCN20000 traz 924 lesões de melanoma (a v2 treinou com 428) e não tem sobreposição de
ID/lesão com HAM10000, ISIC2020 nem MILK10k (auditoria via API do ISIC, ver README).

Desenho:
  - Parte do split_index_v4.csv SEM alterar nada (HAM10000 e MILK10k ficam idênticos, inclusive o
    teste do HAM10000 e as lesões reservadas "test_milk").
  - BCN20000: split por LESÃO (um BCN tem ~4 imagens por lesão), 90% train / 10% val, estratificado
    por diagnóstico. Nada do BCN vai para teste.
  - Checagem de vazamento por phash (dist <= 2) das imagens do BCN contra TODAS as imagens
    dermatoscópicas já usadas (HAM10000, ISIC2020, MILK10k); lesões do BCN com qualquer imagem
    flagrada são descartadas inteiras.

Pré-requisito: scripts/download_isic_images.py bcn20000 e data/raw/candidates/bcn20000.csv.

Uso:
    python scripts/build_v5_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMG_DIR = BASE_DIR / "data" / "raw" / "candidates" / "images" / "bcn20000"
META = BASE_DIR / "data" / "raw" / "candidates" / "bcn20000.csv"
RESIZED_DIR = BASE_DIR / "data" / "processed" / "images_224"
V4 = BASE_DIR / "data" / "processed" / "split_index_v4.csv"
OUT = BASE_DIR / "data" / "processed" / "split_index_v5.csv"

IMG_SIZE = (224, 224)
VAL_FRAC = 0.10
PHASH_MAX_DIST = 2
SEED = 42


def phash_bits(path: Path) -> np.ndarray:
    return imagehash.phash(Image.open(path)).hash.flatten()


def main() -> None:
    m = pd.read_csv(META, low_memory=False)
    m["dx"] = np.where(m["diagnosis_3"].astype(str).str.contains("Nevus"), "nv", "mel")
    m = m[m["isic_id"].map(lambda i: (RAW_IMG_DIR / f"{i}.jpg").exists())].copy()
    print(f"1/5 - {len(m)} imagens do BCN20000 baixadas ({m['lesion_id'].nunique()} lesões; "
          f"{(m['dx'] == 'mel').sum()} mel / {(m['dx'] == 'nv').sum()} nv). Redimensionando...")
    for i in tqdm(m["isic_id"], unit="img"):
        dst = RESIZED_DIR / f"{i}.jpg"
        if not dst.exists():
            with Image.open(RAW_IMG_DIR / f"{i}.jpg") as im:
                im.convert("RGB").resize(IMG_SIZE, Image.BILINEAR).save(dst, "JPEG", quality=95)

    print("2/5 - Checagem de vazamento por phash contra HAM10000, ISIC2020 e MILK10k (dermatoscopia)...")
    v4 = pd.read_csv(V4)
    isic = pd.read_csv(BASE_DIR / "data" / "raw" / "isic2020" / "isic2020_mel_nv_metadata.csv")
    isic_paths = [f"data/processed/images_224/{n}.jpg" for n in isic["image_name"]
                  if (RESIZED_DIR / f"{n}.jpg").exists()]
    used = pd.concat([v4[v4["image_type"] == "dermoscopic"]["path"].str.replace("\\", "/", regex=False),
                      pd.Series(isic_paths)], ignore_index=True).drop_duplicates()
    h_used = np.array([phash_bits(BASE_DIR / p) for p in tqdm(used, unit="img")])
    leaky = set()
    for row in tqdm(list(m.itertuples()), unit="img"):
        h = phash_bits(RESIZED_DIR / f"{row.isic_id}.jpg")
        if (h_used != h).sum(axis=1).min() <= PHASH_MAX_DIST:
            leaky.add(row.lesion_id)
    print(f"      {len(leaky)} lesões do BCN20000 com imagem quase idêntica a uma já usada -> descartadas.")
    m = m[~m["lesion_id"].isin(leaky)].copy()

    print("3/5 - Split por lesão do BCN20000 (train/val)...")
    les = m.drop_duplicates("lesion_id")[["lesion_id", "dx"]].set_index("lesion_id")
    _, idx_val = train_test_split(les.index, test_size=VAL_FRAC, stratify=les["dx"], random_state=SEED)
    m["split"] = np.where(m["lesion_id"].isin(set(idx_val)), "val", "train")

    bcn = pd.DataFrame({
        "image_id": m["isic_id"], "lesion_id": m["lesion_id"],
        "path": [f"data/processed/images_224/{i}.jpg" for i in m["isic_id"]],
        "split": m["split"], "dx": m["dx"], "dx_type": "bcn20000", "is_augmented": False,
        "source_image_id": m["isic_id"], "dataset": "bcn20000",
        "image_type": "dermoscopic", "skin_tone_class": np.nan,
    })

    print("4/5 - Combinando com o manifesto v4 (inalterado)...")
    combined = pd.concat([v4, bcn], ignore_index=True)

    print("5/5 - Validando e salvando...")
    assert (combined.groupby("lesion_id")["split"].nunique() > 1).sum() == 0, "lesão em mais de um split"
    for s in ("val", "test", "test_milk"):
        assert set(v4[v4["split"] == s]["image_id"]) == set(
            combined[(combined["dataset"] != "bcn20000") & (combined["split"] == s)]["image_id"]), f"split {s} mudou"
    print("      Validação OK: nenhuma lesão em dois splits; HAM10000/MILK10k idênticos aos da v4.")
    combined.to_csv(OUT, index=False)

    print(f"\nManifesto v5: {OUT}")
    print("\nPor split e classe:\n", combined.groupby(["split", "dx"]).size().unstack())
    print("\nPor split e fonte:\n", combined.groupby(["split", "dataset"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
