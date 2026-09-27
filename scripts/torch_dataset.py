"""
Dataset PyTorch reutilizável para o classificador mel vs nv (HAM10000).

Encapsula duas decisões importantes documentadas na Sprint 1
(ver scripts/preprocess.py e scripts/split_dataset.py):

1. Imagens aumentadas (is_augmented=True) só devem ser usadas em treino. Usar augmentation
   em val/teste mediria a performance do modelo sobre dado sintético, não sobre a distribuição
   real que ele vai encontrar em produção — por isso `build_datasets()` filtra isso
   automaticamente por split.
2. O split já foi feito por LESÃO (não por imagem) — este módulo não precisa se preocupar com
   isso de novo, só consome a coluna `split` do índice.

Uso:
    from torch_dataset import build_datasets, build_dataloaders

    datasets = build_datasets(base_dir=Path("."))
    loaders = build_dataloaders(datasets, batch_size=32)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_MAP = {"nv": 0, "mel": 1}
CLASS_NAMES = ["nv", "mel"]


def build_transforms(train: bool, strong_aug: bool = False) -> transforms.Compose:
    """Imagens já vêm em 224x224 da Sprint 1 — aqui só normaliza.

    strong_aug=True (treino): augmentation ALEATÓRIO a cada época (rotação livre, flips,
    recorte, brilho/contraste leves). Lesões de pele não têm orientação canônica, então
    rotação/flip preservam o rótulo. Deliberadamente SEM alterar matiz/saturação: a cor é
    sinal diagnóstico do melanoma e é também o atributo que a análise de fairness vai olhar.
    """
    if train and strong_aug:
        return transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MelanomaDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, base_dir: Path, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.base_dir = Path(base_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # "path" foi gravado no Windows (separador \) - normaliza pra rodar em Linux/Kaggle também
        img_path = self.base_dir / row["path"].replace("\\", "/")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = LABEL_MAP[row["dx"]]
        return img, label


def load_split_manifest(base_dir: Path, split_index_path: str | None = None) -> pd.DataFrame:
    base_dir = Path(base_dir)
    path = Path(split_index_path) if split_index_path else base_dir / "data" / "processed" / "split_index.csv"
    df = pd.read_csv(path)
    return df


def build_datasets(base_dir: Path, split_index_path: str | None = None,
                    limit_per_split: int | None = None,
                    use_precomputed_aug: bool = True,
                    strong_aug: bool = False) -> dict[str, MelanomaDataset]:
    """limit_per_split: usar um número pequeno (ex: 20) só para smoke test em CPU local —
    NÃO usar isso pra treino de verdade (isso é só validação de que o pipeline roda sem erro).

    use_precomputed_aug=False: o treino usa só as imagens ORIGINAIS (sem as 5 cópias fixas
    geradas na Sprint 1). Com strong_aug=True cada época vê variações novas, o que evita o
    overfitting por decorar cópias repetidas (visto no treino: recall 0,99 no treino vs 0,5 na val).
    """
    manifest = load_split_manifest(base_dir, split_index_path)

    datasets = {}
    for split in ("train", "val", "test"):
        split_df = manifest[manifest["split"] == split].copy()
        if split != "train" or not use_precomputed_aug:
            # val/test nunca usam imagens aumentadas - ver docstring do módulo
            split_df = split_df[~split_df["is_augmented"]]
        if limit_per_split:
            per_class = max(1, limit_per_split // 2)
            parts = [group.head(per_class) for _, group in split_df.groupby("dx")]
            split_df = pd.concat(parts, ignore_index=True)
        transform = build_transforms(train=(split == "train"), strong_aug=strong_aug)
        datasets[split] = MelanomaDataset(split_df, base_dir, transform=transform)
    return datasets


def build_dataloaders(datasets: dict[str, MelanomaDataset], batch_size: int = 32,
                       num_workers: int = 2) -> dict[str, DataLoader]:
    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return loaders
