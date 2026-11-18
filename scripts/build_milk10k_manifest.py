"""
v4 — adiciona o MILK10k (mel/nv, foto clínica + dermatoscopia da mesma lesão) ao treino.

Motivo: a Sprint 3 mostrou que o gargalo é a troca de modalidade (treino só dermatoscópico), e o
MILK10k é o único dataset aberto com muito melanoma em foto clínica biopsiada. Ver
scripts/evaluate_milk10k.py.

Desenho (mantém o que a v2 tem de comparável):
  - HAM10000: linhas ORIGINAIS com o mesmo split train/val/test da v2 (teste idêntico ->
    qualquer mudança de AUC no HAM10000 é atribuível ao dado novo).
  - MILK10k: split por LESÃO (as duas fotos de uma lesão ficam juntas), estratificado por
    diagnóstico x faixa de tom de pele: 60% train, 10% val, 30% "test_milk". O notebook de treino
    ignora "test_milk"; ele serve para comparar v2 x v4 em dados que nenhum dos dois viu
    (scripts/evaluate_milk10k.py).
  - Checagem de vazamento: lesões do MILK10k cuja dermatoscopia é quase idêntica (phash) a uma
    imagem do HAM10000 ou do ISIC2020 são descartadas, para não contaminar o teste do HAM10000.

Pré-requisito: data/raw/milk10k/{milk10k.zip,training_gt.csv,training_input.csv}.

Uso:
    python scripts/build_milk10k_manifest.py
"""

from __future__ import annotations

import sys
import zipfile
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
RAW_DIR = BASE_DIR / "data" / "raw" / "milk10k"
RESIZED_DIR = BASE_DIR / "data" / "processed" / "images_224"
HAM_SPLIT_INDEX = BASE_DIR / "data" / "processed" / "split_index.csv"
ISIC_METADATA = BASE_DIR / "data" / "raw" / "isic2020" / "isic2020_mel_nv_metadata.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "split_index_v4.csv"

IMG_SIZE = (224, 224)
VAL_FRAC, TEST_FRAC = 0.10, 0.30
PHASH_MAX_DIST = 2  # bem mais estrito que o 4 da Sprint 1, que dava falso positivo em dermatoscopia
SEED = 42


def tone_group(t: int) -> str:
    return "escuro" if t <= 2 else ("medio" if t == 3 else "claro")


def resize_all(df: pd.DataFrame) -> None:
    faltando = [r for r in df.itertuples() if not (RESIZED_DIR / f"{r.isic_id}.jpg").exists()]
    if not faltando:
        print(f"      {len(df)} imagens já redimensionadas, pulando.")
        return
    zf = zipfile.ZipFile(RAW_DIR / "milk10k.zip")
    by_id = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    for r in tqdm(faltando, unit="img"):
        with zf.open(by_id[r.isic_id]) as f:
            Image.open(f).convert("RGB").resize(IMG_SIZE, Image.BILINEAR).save(
                RESIZED_DIR / f"{r.isic_id}.jpg", "JPEG", quality=95)


def phash_bits(path: Path) -> np.ndarray:
    return imagehash.phash(Image.open(path)).hash.flatten()


def find_leaky_lesions(milk: pd.DataFrame, others: pd.DataFrame) -> set[str]:
    derm = milk[milk["image_type"] == "dermoscopic"]
    print(f"      phash: {len(derm)} dermatoscopias do MILK10k x {len(others)} imagens HAM10000/ISIC2020...")
    h_milk = np.array([phash_bits(RESIZED_DIR / f"{i}.jpg") for i in tqdm(derm["isic_id"], unit="img")])
    h_oth = np.array([phash_bits(BASE_DIR / p.replace("\\", "/")) for p in tqdm(others["path"], unit="img")])
    leaky = set()
    for i in range(len(derm)):
        dist = (h_oth != h_milk[i]).sum(axis=1)
        if dist.min() <= PHASH_MAX_DIST:
            leaky.add(derm.iloc[i]["lesion_id"])
    return leaky


def main() -> None:
    gt = pd.read_csv(RAW_DIR / "training_gt.csv")
    inp = pd.read_csv(RAW_DIR / "training_input.csv")
    gt["dx"] = gt[[c for c in gt.columns if c != "lesion_id"]].idxmax(axis=1).str.lower()
    milk = inp.merge(gt[["lesion_id", "dx"]], on="lesion_id")
    milk = milk[milk["dx"].isin(["mel", "nv"])].copy()
    milk["image_type"] = np.where(milk["image_type"].str.contains("dermoscopic"), "dermoscopic", "clinical")
    print(f"1/5 - {len(milk)} imagens / {milk['lesion_id'].nunique()} lesões mel+nv no MILK10k. Redimensionando...")
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    resize_all(milk)

    print("2/5 - Checagem de vazamento contra HAM10000 e ISIC2020 (phash)...")
    ham = pd.read_csv(HAM_SPLIT_INDEX)
    ham_original = ham[~ham["is_augmented"]].copy()
    ham_original["dataset"] = "ham10000"
    isic = pd.read_csv(ISIC_METADATA)
    isic_paths = pd.DataFrame({"path": [str((RESIZED_DIR / f"{n}.jpg").relative_to(BASE_DIR)) for n in isic["image_name"]
                                        if (RESIZED_DIR / f"{n}.jpg").exists()]})
    others = pd.concat([ham_original[["path"]], isic_paths], ignore_index=True)
    leaky = find_leaky_lesions(milk, others)
    print(f"      {len(leaky)} lesões do MILK10k parecem duplicatas de imagens já usadas -> descartadas.")
    milk = milk[~milk["lesion_id"].isin(leaky)].copy()

    print("3/5 - Split por lesão do MILK10k (train/val/test_milk)...")
    les = milk.drop_duplicates("lesion_id")[["lesion_id", "dx", "skin_tone_class"]].copy()
    les["strata"] = les["dx"] + "_" + les["skin_tone_class"].astype(int).map(tone_group)
    idx_trval, idx_test = train_test_split(les.index, test_size=TEST_FRAC, stratify=les["strata"], random_state=SEED)
    sub = les.loc[idx_trval]
    idx_train, idx_val = train_test_split(sub.index, test_size=VAL_FRAC / (1 - TEST_FRAC),
                                          stratify=sub["strata"], random_state=SEED)
    les["split"] = "train"
    les.loc[idx_val, "split"] = "val"
    les.loc[idx_test, "split"] = "test_milk"
    milk = milk.merge(les[["lesion_id", "split"]], on="lesion_id")

    milk_out = pd.DataFrame({
        "image_id": milk["isic_id"], "lesion_id": milk["lesion_id"],
        "path": [str((RESIZED_DIR / f"{i}.jpg").relative_to(BASE_DIR)) for i in milk["isic_id"]],
        "split": milk["split"], "dx": milk["dx"], "dx_type": "milk10k",
        "is_augmented": False, "source_image_id": milk["isic_id"], "dataset": "milk10k",
        "image_type": milk["image_type"], "skin_tone_class": milk["skin_tone_class"],
    })

    print("4/5 - Combinando com HAM10000 original (mesmo split da v2)...")
    cols = ["image_id", "lesion_id", "path", "split", "dx", "dx_type", "is_augmented", "source_image_id", "dataset"]
    ham_out = ham_original[cols].copy()
    ham_out["image_type"] = "dermoscopic"
    ham_out["skin_tone_class"] = np.nan
    combined = pd.concat([ham_out, milk_out], ignore_index=True)

    print("5/5 - Validando e salvando...")
    vazando = combined.groupby("lesion_id")["split"].nunique()
    assert (vazando > 1).sum() == 0, "lesões aparecendo em mais de um split"
    for s in ("val", "test"):
        antes = set(ham_original[ham_original["split"] == s]["image_id"])
        depois = set(combined[(combined["dataset"] == "ham10000") & (combined["split"] == s)]["image_id"])
        assert antes == depois, f"split '{s}' do HAM10000 mudou"
    print("      Validação OK: nenhuma lesão em dois splits; val/test do HAM10000 idênticos aos da v2.")
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nManifesto v4: {OUT_PATH}")
    print("\nPor split e classe:\n", combined.groupby(["split", "dx"]).size().unstack())
    print("\nPor split, fonte e modalidade:\n", combined.groupby(["split", "dataset", "image_type"]).size().unstack(fill_value=0))
    print("\nMILK10k por split, classe e faixa de tom (lesões):")
    print(les.assign(tom=les["skin_tone_class"].astype(int).map(tone_group)).groupby(["split", "dx", "tom"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
