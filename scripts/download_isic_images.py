"""
Baixa as imagens de coleções do ISIC Archive listadas em data/raw/candidates/<nome>.csv
(geradas a partir da API do ISIC: só mel/nv) para data/raw/candidates/images/<nome>/.
Pula o que já foi baixado (pode ser interrompido e retomado).

Uso:
    python scripts/download_isic_images.py bcn20000 hiba251 consec_mel2020_216
"""

from __future__ import annotations

import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CAND_DIR = BASE_DIR / "data" / "raw" / "candidates"
URL = "https://isic-archive.s3.amazonaws.com/images/{}.jpg"
WORKERS = 12


def fetch(isic_id: str, dst: Path) -> bool:
    for tentativa in range(4):
        try:
            with urllib.request.urlopen(URL.format(isic_id), timeout=60) as r:
                data = r.read()
            if len(data) < 1000:
                raise ValueError("resposta pequena demais")
            dst.write_bytes(data)
            return True
        except Exception:
            time.sleep(1.5 * (tentativa + 1))
    return False


def main(names: list[str]) -> None:
    for name in names:
        ids = pd.read_csv(CAND_DIR / f"{name}.csv", usecols=["isic_id"], low_memory=False)["isic_id"].tolist()
        out = CAND_DIR / "images" / name
        out.mkdir(parents=True, exist_ok=True)
        todo = [i for i in ids if not (out / f"{i}.jpg").exists()]
        print(f"{name}: {len(ids)} imagens, {len(todo)} a baixar", flush=True)
        falhas, feitos = [], 0
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(fetch, i, out / f"{i}.jpg"): i for i in todo}
            for f in as_completed(futs):
                feitos += 1
                if not f.result():
                    falhas.append(futs[f])
                if feitos % 500 == 0:
                    print(f"  {name}: {feitos}/{len(todo)}", flush=True)
        print(f"{name}: concluído, {len(falhas)} falhas {falhas[:5]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
