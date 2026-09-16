"""
TCC-02 — Pré-processamento das imagens do HAM10000 (melanoma vs. nevo).

Dataset: HAM10000 (Tschandl et al. 2018), baixado via Harvard Dataverse
(doi:10.7910/DVN/DBW86T), licença CC-BY-NC. Contém 10.015 imagens dermatoscópicas em
7 categorias; este projeto foca em `mel` (melanoma) vs `nv` (nevo/pinta benigna) — o
diferencial clínico clássico e o par mais relevante para triagem de câncer de pele
(ver notebooks/01_eda.ipynb para a justificativa completa da escolha de dataset).

IMPORTANTE — por que HAM10000 não tem tom de pele: este dataset não tem anotação de
Fitzpatrick. A análise de fairness por tom de pele deste TCC usa o dataset DDI (Stanford,
Diverse Dermatology Images) como benchmark de avaliação separado — ver scripts/prepare_ddi.py
e a Seção correspondente do notebook. Este script cuida só do treino/val/teste "internos"
do classificador mel vs nv.

Decisões técnicas:
- Resolução 224x224 (padrão ResNet/EfficientNet pré-treinadas em ImageNet).
- Normalização ImageNet aplicada on-the-fly no data loader (não em disco) — mesma
  lógica documentada na versão anterior deste script (ver archive_fitzpatrick17k_v1/).
- Cada LESÃO (lesion_id) pode ter mais de uma foto (1.801 lesões mel+nv têm 2+ imagens).
  A deduplicação de qualidade (corrompidas/duplicatas) e o augmentation tratam cada
  imagem individualmente, mas o SPLIT (TCC-03) agrupa por lesion_id — nunca por imagem —
  para não vazar a mesma lesão entre treino e teste.
- Augmentation balanceia POR CLASSE (mel é 1.113 imagens vs 6.705 de nv, ~1:6), não por
  tom de pele (não aplicável aqui).

Uso:
    python scripts/preprocess.py
    python scripts/preprocess.py --target-mel-mult 4 --max-mult 6
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import imagehash
except ImportError:  # pragma: no cover
    imagehash = None

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "ham10000"
METADATA_PATH = RAW_DIR / "HAM10000_metadata.tab"
IMAGES_DIR = RAW_DIR / "images"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESIZED_DIR = PROCESSED_DIR / "images_224"
MANIFEST_PATH = PROCESSED_DIR / "preprocessed_manifest.csv"

IMG_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ["mel", "nv"]


# ---------------------------------------------------------------------------
# Qualidade: corrompidas e duplicatas
# ---------------------------------------------------------------------------

def find_corrupted_images(images_dir: Path, image_ids: set[str]) -> set[str]:
    corrompidas = set()
    for image_id in image_ids:
        path = images_dir / f"{image_id}.jpg"
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrompidas.add(image_id)
    return corrompidas


def find_exact_duplicates(images_dir: Path, image_ids: set[str]) -> set[str]:
    seen: dict[str, str] = {}
    duplicates = set()
    for image_id in sorted(image_ids):
        path = images_dir / f"{image_id}.jpg"
        content_hash = hashlib.md5(path.read_bytes()).hexdigest()
        if content_hash in seen:
            duplicates.add(image_id)
        else:
            seen[content_hash] = image_id
    return duplicates


def find_near_duplicates(images_dir: Path, image_ids: set[str], hamming_threshold: int = 4) -> set[str]:
    """ATENÇÃO: no HAM10000, fotos da MESMA lesão são naturalmente quase-idênticas
    (mesma pinta, ângulos/zoom levemente diferentes) — isso é esperado e NÃO deve ser
    removido como 'duplicata', é dado real (múltiplas fotos da mesma lesão). Por isso,
    aqui só sinalizamos pares quase-idênticos que pertencem a LESÕES DIFERENTES
    (lesion_id distinto), que aí sim seriam um problema real de qualidade do dataset.
    """
    if imagehash is None:
        print("aviso: pacote 'imagehash' não instalado, pulando deduplicação visual.")
        return set()
    meta = pd.read_csv(METADATA_PATH, sep="\t", quotechar='"').set_index("image_id")["lesion_id"]

    seen: dict = {}
    duplicates = set()
    for image_id in sorted(image_ids):
        path = images_dir / f"{image_id}.jpg"
        with Image.open(path) as img:
            ph = imagehash.phash(img)
        for other_hash, other_id in seen.items():
            if ph - other_hash <= hamming_threshold and meta.get(other_id) != meta.get(image_id):
                duplicates.add(image_id)
                break
        seen[ph] = image_id
    return duplicates


# ---------------------------------------------------------------------------
# Resize + normalização
# ---------------------------------------------------------------------------

def load_and_resize(path: Path, size: tuple[int, int] = IMG_SIZE) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return img.resize(size, Image.BILINEAR)


def normalize_image_array(img: Image.Image) -> np.ndarray:
    """Converte a imagem (0-255) num array float32 normalizado no padrão ImageNet.
    Usar no data loader de treino (Sprint 2), não ao salvar em disco."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    rotation_range_deg: float = 20.0
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    hflip_prob: float = 0.5


def augment_image(img: Image.Image, cfg: AugmentConfig, rng: random.Random) -> Image.Image:
    out = img
    angle = rng.uniform(-cfg.rotation_range_deg, cfg.rotation_range_deg)
    out = out.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    if rng.random() < cfg.hflip_prob:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(*cfg.brightness_range))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(*cfg.contrast_range))
    return out


def build_balance_plan(df: pd.DataFrame, class_col: str, max_mult: int) -> dict:
    """Copias aumentadas por imagem, por classe, pra aproximar mel do volume de nv."""
    counts = df[class_col].value_counts()
    target = counts.max()
    plan = {}
    for cls, current_count in counts.items():
        if current_count >= target:
            plan[cls] = 0
            continue
        needed = target - current_count
        rows_in_class = df[df[class_col] == cls]
        per_image = min(max_mult, -(-needed // len(rows_in_class)))
        plan[cls] = per_image
    return plan


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-mult", type=int, default=5,
                         help="teto de cópias aumentadas por imagem-base (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-near-duplicates", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 - Carregando metadados e filtrando mel vs nv...")
    meta = pd.read_csv(METADATA_PATH, sep="\t", quotechar='"')
    df = meta[meta["dx"].isin(CLASSES)].copy()
    print(f"      {len(df)} imagens ({df['dx'].value_counts().to_dict()})")

    image_ids = set(df["image_id"])

    print("2/5 - Identificando imagens problemáticas (corrompidas / duplicatas)...")
    corrompidas = find_corrupted_images(IMAGES_DIR, image_ids)
    exatas = find_exact_duplicates(IMAGES_DIR, image_ids)
    visuais = set() if args.skip_near_duplicates else find_near_duplicates(IMAGES_DIR, image_ids)
    remover = corrompidas | exatas | visuais
    print(f"      corrompidas={len(corrompidas)} duplicatas_exatas={len(exatas)} "
          f"duplicatas_visuais_entre_lesoes_diferentes={len(visuais)}")
    df = df[~df["image_id"].isin(remover)].copy()
    print(f"      {len(df)} imagens restantes após limpeza")

    print("3/5 - Redimensionando imagens originais para 224x224...")
    manifest_rows = []
    for row in tqdm(df.itertuples(), total=len(df), unit="img"):
        src = IMAGES_DIR / f"{row.image_id}.jpg"
        dst = RESIZED_DIR / f"{row.image_id}.jpg"
        img = load_and_resize(src)
        img.save(dst, "JPEG", quality=95)
        manifest_rows.append({
            "image_id": row.image_id,
            "lesion_id": row.lesion_id,
            "path": str(dst.relative_to(BASE_DIR)),
            "dx": row.dx,
            "dx_type": row.dx_type,
            "is_augmented": False,
            "source_image_id": row.image_id,
        })

    print("4/5 - Planejando e gerando data augmentation (balanceando mel vs nv)...")
    plan = build_balance_plan(df, "dx", args.max_mult)
    print(f"      cópias aumentadas por imagem, por classe: {plan}")

    cfg = AugmentConfig()
    for row in tqdm(df.itertuples(), total=len(df), unit="img"):
        n_copies = plan.get(row.dx, 0)
        if n_copies == 0:
            continue
        src = IMAGES_DIR / f"{row.image_id}.jpg"
        with Image.open(src) as base_img:
            base_img = base_img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
            for i in range(n_copies):
                aug_img = augment_image(base_img, cfg, rng)
                aug_id = f"{row.image_id}_aug{i}"
                dst = RESIZED_DIR / f"{aug_id}.jpg"
                aug_img.save(dst, "JPEG", quality=95)
                manifest_rows.append({
                    "image_id": aug_id,
                    "lesion_id": row.lesion_id,
                    "path": str(dst.relative_to(BASE_DIR)),
                    "dx": row.dx,
                    "dx_type": row.dx_type,
                    "is_augmented": True,
                    "source_image_id": row.image_id,
                })

    print("5/5 - Salvando manifesto...")
    manifest = pd.DataFrame(manifest_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    n_orig = (~manifest["is_augmented"]).sum()
    n_aug = manifest["is_augmented"].sum()
    print(f"\nConcluído: {n_orig} imagens originais + {n_aug} aumentadas = {len(manifest)} no manifesto.")
    print(f"Manifesto: {MANIFEST_PATH}")
    print(f"Imagens redimensionadas em: {RESIZED_DIR}")
    print("\nDistribuição final por classe (originais + aumentadas):")
    print(manifest["dx"].value_counts())
    print("\nLEMBRETE: augmentation deve ser usado só em split=='train' (ver TCC-03/split_dataset.py).")


if __name__ == "__main__":
    main()
