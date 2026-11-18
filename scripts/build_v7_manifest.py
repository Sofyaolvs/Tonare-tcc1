"""
v7 — v6 + fotos clínicas do Fitzpatrick17k dos tons I-IV no treino; tons V-VI ficam SÓ para teste.

Desenho:
  - Parte do split_index_v6.csv sem alterar val, test e test_milk.
  - Fitzpatrick17k: melanoma ("melanoma", "superficial spreading melanoma ssm", "malignant melanoma")
    vs nevo melanocítico ("nevocytic nevus", "halo nevus", "congenital nevus"), escala Fitzpatrick 1-4.
    FORA do treino: tons 5-6 (teste), tom não anotado e "lentigo maligna" (todas do Atlas Dermatológico,
    fonte diferente — viraria atalho de fonte).
  - O dataset não tem id de lesão: imagens quase idênticas (phash <= 4) são agrupadas em "lesões" e o
    split train/val (90/10) é por grupo.
  - Vazamento: imagens candidatas com phash <= 2 de qualquer foto clínica dos testes externos
    (MILK10k reservado, PAD-UFES-20, HIBA clínico, DDI), ou <= 3 de qualquer imagem do Fitzpatrick 5-6,
    são descartadas.
  - Imagens do Fitzpatrick no treino aparecem FITZ_COPIES vezes (mesma ideia da v6; augmentation aleatório).
  - Gera também um zip só com o que precisa subir no Kaggle (imagens novas + manifesto).

Uso:
    python scripts/build_v7_manifest.py [caminho/fitzpatrick17k.zip]
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
PROC = BASE_DIR / "data" / "processed"
RAW_FZ = BASE_DIR / "data" / "raw" / "fitzpatrick17k"
ZIP_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "fitzpatrick17k.zip"
V6 = PROC / "split_index_v6.csv"
OUT = PROC / "split_index_v7.csv"
UPLOAD_ZIP = Path.home() / "Downloads" / "tonare_v7_upload.zip"

MEL = ["melanoma", "superficial spreading melanoma ssm", "malignant melanoma"]
NEV = ["nevocytic nevus", "halo nevus", "congenital nevus"]
FITZ_COPIES = 2
VAL_FRAC = 0.10
DIST_CLUSTER, DIST_EXTERNAL, DIST_DARK = 4, 2, 3
SEED = 42


def ph(img: Image.Image) -> np.ndarray:
    return imagehash.phash(img.convert("RGB").resize((224, 224), Image.BILINEAR)).hash.flatten()


def ph_path(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return ph(im)


def min_dist(h: np.ndarray, pool: np.ndarray) -> int:
    return int((pool != h).sum(axis=1).min()) if len(pool) else 99


def ensure_raw(ids: list[str]) -> Path:
    out = RAW_FZ / "images"
    out.mkdir(parents=True, exist_ok=True)
    faltam = [i for i in ids if not (out / f"{i}.jpg").exists()]
    if faltam:
        with zipfile.ZipFile(ZIP_PATH) as z:
            by = {Path(n).name: n for n in z.namelist() if n.endswith(".jpg")}
            for i in faltam:
                (out / f"{i}.jpg").write_bytes(z.read(by[f"{i}.jpg"]))
    return out


def external_clinical_hashes() -> np.ndarray:
    hs = []
    v6 = pd.read_csv(V6)
    mk = v6[(v6["split"] == "test_milk") & (v6["image_type"] == "clinical")]
    hs += [ph_path(BASE_DIR / p.replace("\\", "/")) for p in tqdm(mk["path"], desc="MILK reservado", unit="img")]
    pad = pd.read_csv(BASE_DIR / "data" / "raw" / "padufes20" / "metadata.csv")
    pad = pad[pad["diagnostic"].isin(["MEL", "NEV"])]
    hs += [ph_path(BASE_DIR / "data" / "raw" / "padufes20" / "images" / i) for i in tqdm(pad["img_id"], desc="PAD-UFES-20", unit="img")]
    hb = pd.read_csv(BASE_DIR / "data" / "raw" / "candidates" / "hiba251.csv", low_memory=False)
    hb = hb[hb["acq_image_type"].astype(str).str.contains("clinical")]
    hs += [ph_path(BASE_DIR / "data" / "raw" / "candidates" / "images" / "hiba251" / f"{i}.jpg") for i in tqdm(hb["isic_id"], desc="HIBA clínico", unit="img")]
    dd = pd.read_csv(PROC / "ddi" / "ddi_manifest.csv")
    hs += [ph_path(BASE_DIR / p.replace("\\", "/")) for p in tqdm(dd["path"], desc="DDI", unit="img")]
    return np.array(hs)


def main() -> None:
    fz = pd.read_csv(RAW_FZ / "fitzpatrick17k.csv")
    fz = fz[fz["label"].isin(MEL + NEV)].copy()
    fz["dx"] = np.where(fz["label"].isin(MEL), "mel", "nv")
    cand = fz[fz["fitzpatrick_scale"].between(1, 4)].copy()
    dark = fz[fz["fitzpatrick_scale"].isin([5, 6])]
    print(f"1/6 - Fitzpatrick17k: {len(cand)} candidatas (tons 1-4: {int((cand.dx == 'mel').sum())} mel / "
          f"{int((cand.dx == 'nv').sum())} nv) | tons 5-6 ({len(dark)}) e sem tom ficam de fora do treino.")
    raw = ensure_raw(list(fz["md5hash"]))

    print("2/6 - Hashes das imagens candidatas, do teste escuro e dos testes externos clínicos...")
    h_cand = np.array([ph_path(raw / f"{i}.jpg") for i in tqdm(cand["md5hash"], desc="candidatas", unit="img")])
    h_dark = np.array([ph_path(raw / f"{i}.jpg") for i in tqdm(dark["md5hash"], desc="tons 5-6", unit="img")])
    h_ext = external_clinical_hashes()

    print("3/6 - Descartando possíveis duplicatas de teste...")
    d_ext = np.array([min_dist(h, h_ext) for h in h_cand])
    d_dark = np.array([min_dist(h, h_dark) for h in h_cand])
    drop = (d_ext <= DIST_EXTERNAL) | (d_dark <= DIST_DARK)
    print(f"      {int((d_ext <= DIST_EXTERNAL).sum())} parecidas com testes externos (<= {DIST_EXTERNAL}), "
          f"{int((d_dark <= DIST_DARK).sum())} parecidas com tons 5-6 (<= {DIST_DARK}) -> {int(drop.sum())} descartadas.")
    keep = ~drop
    cand, h_cand = cand[keep].reset_index(drop=True), h_cand[keep]

    print("4/6 - Agrupando quase-duplicatas (phash <= 4) e dividindo por grupo...")
    parent = list(range(len(cand)))
    find = lambda x: x if parent[x] == x else find(parent[x])
    for i in range(len(cand)):
        for j in np.where((h_cand[i + 1:] != h_cand[i]).sum(axis=1) <= DIST_CLUSTER)[0] + i + 1:
            parent[find(j)] = find(i)
    cand["grp"] = [f"FZ_{find(i)}" for i in range(len(cand))]
    grp = cand.groupby("grp")["dx"].agg(lambda s: s.value_counts().index[0])
    print(f"      {len(cand)} imagens em {len(grp)} grupos.")
    _, g_val = train_test_split(grp.index, test_size=VAL_FRAC, stratify=grp.values, random_state=SEED)
    cand["split"] = np.where(cand["grp"].isin(set(g_val)), "val", "train")

    print("5/6 - Redimensionando (224x224 bilinear + JPEG q95) e montando o manifesto...")
    RES = PROC / "images_224"
    for i in tqdm(cand["md5hash"], unit="img"):
        dst = RES / f"fz_{i}.jpg"
        if not dst.exists():
            with Image.open(raw / f"{i}.jpg") as im:
                im.convert("RGB").resize((224, 224), Image.BILINEAR).save(dst, "JPEG", quality=95)
    fz_rows = pd.DataFrame({
        "image_id": "fz_" + cand["md5hash"], "lesion_id": cand["grp"],
        "path": [f"data/processed/images_224/fz_{i}.jpg" for i in cand["md5hash"]],
        "split": cand["split"], "dx": cand["dx"], "dx_type": "fitzpatrick17k_atlas", "is_augmented": False,
        "source_image_id": cand["md5hash"], "dataset": "fitzpatrick17k", "image_type": "clinical",
        "skin_tone_class": np.nan,
    })
    v6 = pd.read_csv(V6)
    extra = pd.concat([fz_rows[fz_rows["split"] == "train"]] * (FITZ_COPIES - 1), ignore_index=True)
    v7 = pd.concat([v6, fz_rows, extra], ignore_index=True)

    print("6/6 - Validando e salvando...")
    for s in ("val", "test", "test_milk"):
        assert v6[v6["split"] == s].reset_index(drop=True).equals(
            v7[(v7["split"] == s) & (v7["dataset"] != "fitzpatrick17k")].reset_index(drop=True)), f"split {s} mudou"
    assert (v7.groupby("lesion_id")["split"].nunique() > 1).sum() == 0, "lesão em mais de um split"
    assert not set(dark["md5hash"]) & set(fz_rows["source_image_id"]), "imagem de tom 5-6 entrou no manifesto"
    print("      Validação OK: val/test/test_milk iguais aos da v6; nenhuma lesão em dois splits; tons 5-6 fora.")
    v7.to_csv(OUT, index=False)

    with zipfile.ZipFile(UPLOAD_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUT, "processed/split_index_v7.csv")
        for i in fz_rows["source_image_id"]:
            z.write(RES / f"fz_{i}.jpg", f"processed/images_224/fz_{i}.jpg")

    tr = v7[v7["split"] == "train"]
    print(f"\nManifesto v7: {OUT}\nZip para o Kaggle: {UPLOAD_ZIP} ({UPLOAD_ZIP.stat().st_size // 2**20} MB)")
    print(f"Treino: {len(tr)} imagens (v6: {int((v6['split'] == 'train').sum())}) | classes: {tr['dx'].value_counts().to_dict()}")
    print("Fração do treino por fonte/modalidade:")
    print((tr.groupby(["dataset", "image_type"]).size() / len(tr)).round(3).to_string())
    print("Validação por fonte:", v7[v7["split"] == "val"].groupby("dataset").size().to_dict())


if __name__ == "__main__":
    main()
