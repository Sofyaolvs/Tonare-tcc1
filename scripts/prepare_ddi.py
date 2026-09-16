"""
Preparação do DDI (Diverse Dermatology Images, Stanford) — benchmark de fairness por
tom de pele deste TCC.

O DDI NÃO faz parte do treino: é um conjunto de avaliação separado, usado só depois que o
classificador mel vs nv for treinado no HAM10000 (ver scripts/preprocess.py e
scripts/split_dataset.py). A ideia é medir sensibilidade/especificidade do modelo treinado
separadamente nas 3 faixas de tom de pele do DDI, para responder a pergunta central de
fairness do TCC: o modelo erra mais em pele escura?

Como conseguir os dados (não é possível automatizar — exige registro pessoal):
    1. Acesse https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965
    2. Registre-se e aceite o Research Use Agreement (uso não-comercial, pesquisa)
    3. Baixe e extraia o conteúdo em: data/raw/ddi/
       Estrutura esperada:
           data/raw/ddi/ddi_metadata.csv
           data/raw/ddi/images/*.png  (ou .jpg, dependendo do pacote baixado)

Uso:
    python scripts/prepare_ddi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
DDI_RAW_DIR = BASE_DIR / "data" / "raw" / "ddi"
DDI_METADATA_PATH = DDI_RAW_DIR / "ddi_metadata.csv"
DDI_IMAGES_DIR = DDI_RAW_DIR / "images"

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ddi"
RESIZED_DIR = PROCESSED_DIR / "images_224"
MANIFEST_PATH = PROCESSED_DIR / "ddi_manifest.csv"

IMG_SIZE = (224, 224)

SKIN_TONE_LABELS = {
    12: "claro (FST I-II)",
    34: "médio (FST III-IV)",
    56: "escuro (FST V-VI)",
}


def find_image_file(ddi_file: str) -> Path | None:
    """DDI_file pode vir com ou sem extensão, e o pacote baixado pode trazer .png ou .jpg."""
    stem = Path(ddi_file).stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = DDI_IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    if not DDI_METADATA_PATH.exists():
        print(f"ERRO: {DDI_METADATA_PATH} não encontrado.")
        print("\nO DDI exige registro pessoal (Research Use Agreement da Stanford) — não dá pra")
        print("automatizar o download. Ver instruções no topo deste script (docstring) para os")
        print("passos de cadastro, e coloque os arquivos baixados em data/raw/ddi/ antes de rodar.")
        sys.exit(1)

    print("1/3 - Carregando metadados do DDI...")
    meta = pd.read_csv(DDI_METADATA_PATH)
    print(f"      Colunas encontradas: {list(meta.columns)}")

    # nomes de coluna variam um pouco entre pacotes do DDI distribuídos - normaliza
    col_file = next((c for c in meta.columns if c.lower() in ("ddi_file", "file", "filename")), None)
    col_skin = next((c for c in meta.columns if "skin_tone" in c.lower() or "fitzpatrick" in c.lower()), None)
    col_malignant = next((c for c in meta.columns if "malignan" in c.lower()), None)

    if not all([col_file, col_skin, col_malignant]):
        print(f"\nAVISO: não consegui identificar automaticamente todas as colunas esperadas.")
        print(f"        arquivo={col_file}  tom_de_pele={col_skin}  malignidade={col_malignant}")
        print(f"        Colunas disponíveis: {list(meta.columns)}")
        print("        Ajuste os nomes de coluna no topo da função main() deste script e rode de novo.")
        sys.exit(1)

    meta = meta.rename(columns={col_file: "ddi_file", col_skin: "skin_tone", col_malignant: "malignant"})
    meta["skin_tone_band"] = meta["skin_tone"].map(SKIN_TONE_LABELS).fillna(meta["skin_tone"].astype(str))

    print(f"\n      Total de imagens no metadata: {len(meta)}")
    print("\n      Cruzamento malignidade x tom de pele (dataset completo, antes de checar arquivos):")
    print(pd.crosstab(meta["malignant"], meta["skin_tone_band"]))

    print("\n2/3 - Localizando arquivos de imagem e redimensionando para 224x224...")
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    faltando = []
    for row in meta.itertuples():
        src = find_image_file(row.ddi_file)
        if src is None:
            faltando.append(row.ddi_file)
            continue
        dst = RESIZED_DIR / f"{src.stem}.jpg"
        with Image.open(src) as img:
            img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
            img.save(dst, "JPEG", quality=95)
        manifest_rows.append({
            "ddi_file": row.ddi_file,
            "path": str(dst.relative_to(BASE_DIR)),
            "malignant": row.malignant,
            "skin_tone": row.skin_tone,
            "skin_tone_band": row.skin_tone_band,
        })

    if faltando:
        print(f"      AVISO: {len(faltando)} imagens listadas no CSV não foram encontradas em "
              f"{DDI_IMAGES_DIR} (conferir extensão/nome do arquivo baixado).")

    print("3/3 - Salvando manifesto...")
    manifest = pd.DataFrame(manifest_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"\nConcluído: {len(manifest)} imagens do DDI processadas.")
    print(f"Manifesto: {MANIFEST_PATH}")
    print(f"Imagens redimensionadas em: {RESIZED_DIR}")
    print("\nCruzamento final malignidade x tom de pele (só imagens efetivamente processadas):")
    print(pd.crosstab(manifest["malignant"], manifest["skin_tone_band"]))
    print("\nUso na Sprint 2: rode o classificador treinado no HAM10000 sobre estas imagens e "
          "compare sensibilidade/especificidade entre as 3 faixas de skin_tone_band — esse é o "
          "resultado central de fairness do TCC.")


if __name__ == "__main__":
    main()
