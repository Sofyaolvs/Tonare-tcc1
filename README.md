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

## Sprint 2 — Setup do ambiente e treinamento do modelo base

| Task | Entregável |
|---|---|
| TCC-04 — Treino do modelo base | [`notebooks/02_train_baseline.ipynb`](notebooks/02_train_baseline.ipynb) |
| Dataset PyTorch reutilizável | [`scripts/torch_dataset.py`](scripts/torch_dataset.py) |

**Framework:** PyTorch. **Ambiente de treino:** esta máquina não tem GPU dedicada (só placa
integrada), então o treino de verdade roda no **Kaggle Notebooks** (GPU gratuita). O notebook
detecta o ambiente sozinho (`IS_KAGGLE`): local roda só um smoke test (amostra pequena, CPU,
1 época — prova que o pipeline não tem bug antes de gastar cota de GPU); no Kaggle roda o
treino real.

**Histórico do treino:** a v1 (5 cópias fixas de cada melanoma vindas da Sprint 1, sem early
stopping) deu recall de mel = 0,58 no teste, AUC de validação 0,89 e overfitting claro
(recall de treino 0,99 vs 0,5 na validação). A v2 usa só imagens originais + augmentation
aleatório a cada época, early stopping pela AUC de validação e limiar de decisão escolhido na
validação com meta de sensibilidade.

### Como rodar no Kaggle

1. O Kaggle já vem com PyTorch+GPU (`requirements-train.txt` documenta as versões usadas local).
2. Suba `data/processed/` (pasta `images_224/` + `preprocessed_manifest.csv` +
   `split_index.csv`) como um Kaggle Dataset privado.
3. Crie um notebook, anexe o dataset, ative GPU e Internet (exige celular verificado na conta).
4. Copie as células de `notebooks/02_train_baseline.ipynb`; na Célula 1, no lugar do import,
   cole o conteúdo de `scripts/torch_dataset.py` (import relativo não funciona lá) e ajuste o
   caminho `KAGGLE_PROCESSED` na célula de dados para o seu dataset.

## Estrutura

```
data/raw/ham10000/     metadados + imagens originais do HAM10000 (gitignored)
data/raw/ddi/           metadados + imagens do DDI, após você baixar manualmente (gitignored)
data/processed/         imagens redimensionadas, manifesto, índice de split e checkpoints (gitignored)
notebooks/               EDA (TCC-01) e treino do modelo base (TCC-04)
scripts/                 pré-processamento (TCC-02), split (TCC-03), preparação do DDI e dataset PyTorch (TCC-04)
```
