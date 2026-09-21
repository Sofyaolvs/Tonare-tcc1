# Tonare — TCC

Triagem de **melanoma** por imagem (CNN + transfer learning) com análise de equidade
(fairness) entre tons de pele.

## Estratégia de dados (v2)

Nenhum dataset público sozinho tem volume grande de melanoma **e** anotação de tom de pele
balanceada. Por isso, combinamos dois:

| Dataset | Papel | Por quê |
|---|---|---|
| **[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)** | Treino/val/teste do classificador (`mel` vs `nv`) | 1.113 imagens de melanoma confirmadas — volume real. Sem tom de pele. |
| **[DDI](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965)** (Stanford) | Avaliação de fairness | 656 imagens, biópsia-confirmadas, tom de pele nativo e balanceado em 3 faixas. |

Ver `notebooks/01_eda.ipynb` (Seção "Por que trocamos de dataset") para o raciocínio completo.
A primeira tentativa (Fitzpatrick17k) foi descartada por cobrir 114 condições genéricas (não
só melanoma) e ter 76% dos links de imagem mortos — os arquivos dessa tentativa foram removidos
do projeto.

## Sprint 1 — Preparação dos dados

| Task | Entregável |
|---|---|
| TCC-01 — EDA | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) |
| TCC-02 — Pré-processamento | [`scripts/preprocess.py`](scripts/preprocess.py) |
| TCC-03 — Split treino/val/teste | [`scripts/split_dataset.py`](scripts/split_dataset.py) |
| Preparação do benchmark de fairness | [`scripts/prepare_ddi.py`](scripts/prepare_ddi.py) (pendente: requer registro no DDI, ver abaixo) |

## Como reproduzir

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows

# 1. Baixar HAM10000 (CC-BY-NC, sem login) — metadados + imagens
curl -L -o data/raw/ham10000/HAM10000_metadata.tab https://dataverse.harvard.edu/api/access/datafile/4338392
curl -L -o data/raw/ham10000/HAM10000_images_part_1.zip https://dataverse.harvard.edu/api/access/datafile/3172585
curl -L -o data/raw/ham10000/HAM10000_images_part_2.zip https://dataverse.harvard.edu/api/access/datafile/3172584
# extrair os dois zips em data/raw/ham10000/images/

# 2. Pré-processar (resize 224x224 + limpeza + augmentation) -> TCC-02
python scripts/preprocess.py

# 3. Gerar split 70/15/15 estratificado por classe, agrupado por lesão -> TCC-03
python scripts/split_dataset.py

# 4. Abrir o notebook de EDA -> TCC-01
jupyter notebook notebooks/01_eda.ipynb

# 5. DDI (fairness) — requer registro manual, ver scripts/prepare_ddi.py
python scripts/prepare_ddi.py
```

### Registro no DDI (não automatizável)

1. Acesse o [portal AIMI da Stanford](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965)
2. Cadastre-se (pode usar e-mail institucional) e aceite o Research Use Agreement
3. Baixe e extraia em `data/raw/ddi/` (esperado: `ddi_metadata.csv` + `images/`)
4. Rode `python scripts/prepare_ddi.py`

## Estrutura

```
data/raw/ham10000/     metadados + imagens originais do HAM10000 (gitignored)
data/raw/ddi/           metadados + imagens do DDI, após você baixar manualmente (gitignored)
data/processed/         imagens redimensionadas, manifesto e índice de split (gitignored)
notebooks/               EDA (TCC-01)
scripts/                 pré-processamento (TCC-02), split (TCC-03) e preparação do DDI
```
