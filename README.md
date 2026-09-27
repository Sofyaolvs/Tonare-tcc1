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
| Preparação do benchmark de fairness | [`scripts/prepare_ddi.py`](scripts/prepare_ddi.py) (concluído — ver Sprint 3 abaixo) |

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

**Histórico do treino:**
- **v1** (5 cópias fixas de cada melanoma vindas da Sprint 1, sem early stopping): recall de mel
  0,58 no teste, AUC de validação 0,89, overfitting claro (recall de treino 0,99 vs 0,5 na val).
- **v2** ([`notebooks/02_train_baseline.ipynb`](notebooks/02_train_baseline.ipynb)): só imagens
  originais + augmentation aleatório a cada época, early stopping pela AUC de validação, limiar
  escolhido na validação. Resultado: **AUC de teste 0,922**, sensibilidade 0,885 no limiar 0,21.
- **v3** ([`notebooks/02_train_baseline_v3_combined.ipynb`](notebooks/02_train_baseline_v3_combined.ipynb)):
  soma o **ISIC 2020** (584 melanomas + 5.193 nevos confirmados, sem duplicata com o HAM10000 —
  ver [`scripts/prepare_isic2020.py`](scripts/prepare_isic2020.py) e
  [`scripts/build_combined_manifest.py`](scripts/build_combined_manifest.py)) só no treino/val;
  o teste do HAM10000 é mantido idêntico ao da v2, de propósito, para isolar o efeito de ter
  mais dado de treino. Melanoma no treino: 771 → 1.297 imagens.

### Como rodar no Kaggle

1. O Kaggle já vem com PyTorch+GPU (`requirements-train.txt` documenta as versões usadas local).
2. Suba `data/processed/` (pasta `images_224/` + `preprocessed_manifest.csv` + `split_index.csv`
   + `split_index_combined.csv`, se for rodar a v3) como um Kaggle Dataset privado.
3. Crie um notebook, anexe o dataset, ative GPU e Internet (exige celular verificado na conta).
4. Copie as células do notebook (v2 ou v3); na Célula 1, no lugar do import, cole o conteúdo de
   `scripts/torch_dataset.py` (import relativo não funciona lá) e ajuste o caminho
   `KAGGLE_PROCESSED` na célula de dados para o seu dataset.

Checkpoints resultantes (baixados do Kaggle, não vão pro git — ver `.gitignore`):
`data/processed/model_resnet50_baseline_v2.pt` (referência, recomendado) e `_v3.pt`.

## Sprint 3 — Avaliação de fairness (DDI)

| Task | Entregável |
|---|---|
| Avaliação por tom de pele | [`scripts/evaluate_ddi.py`](scripts/evaluate_ddi.py) |
| Análise controlada por doença | [`scripts/analyze_ddi_by_disease.py`](scripts/analyze_ddi_by_disease.py) |

Roda o checkpoint treinado sobre as 656 imagens do DDI e compara sensibilidade/especificidade/AUC
entre as 3 faixas de tom de pele, com intervalo de confiança por bootstrap. `CHECKPOINT_VERSION`
no topo de cada script escolhe qual modelo avaliar (`v2` por padrão).

**Achado central:** a AUC geral cai de 0,92 (HAM10000) para 0,62 (DDI, v2) — sobretudo por troca
de modalidade (dermatoscopia → foto clínica), não por tom de pele. Uma comparação ingênua por
faixa parece mostrar viés contra pele escura (AUC 0,49), mas 56% dos casos malignos dessa faixa
no DDI são linfoma cutâneo (doença nunca vista no treino) — controlando por doença, a diferença
não se sustenta estatisticamente (só 7 melanomas por faixa no DDI, poder insuficiente).

Tentamos resolver ampliando o treino com o ISIC 2020 (v3, ver Sprint 2 acima): a AUC no HAM10000
não mudou, e a generalização pro DDI **piorou** (AUC 0,55) — mais dado do mesmo tipo de imagem
não resolve a troca de modalidade. Por isso o **v2 é o modelo de referência**, não o v3.

```bash
python scripts/prepare_isic2020.py       # extrai/dedupe o ISIC2020 contra o HAM10000 (opcional, só p/ v3)
python scripts/build_combined_manifest.py  # gera split_index_combined.csv (opcional, só p/ v3)
python scripts/evaluate_ddi.py
python scripts/analyze_ddi_by_disease.py
```

## Interface de demonstração

```bash
python scripts/demo_app.py
```
Abre em `http://127.0.0.1:7860`. Interface local (Gradio) para testar o classificador com uma
foto solta na tela ou um dos 6 casos de exemplo em `data/demo_examples/`. Estilo em
[`scripts/tonare.css`](scripts/tonare.css).
Requer `data/processed/model_resnet50_baseline_v2.pt` (ver Sprint 2).

## Estrutura

```
data/raw/ham10000/       metadados + imagens originais do HAM10000 (gitignored)
data/raw/ddi/            metadados + imagens do DDI, após você baixar manualmente (gitignored)
data/raw/isic2020/       imagens mel/nv extraídas do ISIC2020, dedupe contra o HAM10000 (gitignored)
data/processed/          imagens redimensionadas, manifestos, índices de split e checkpoints (gitignored)
data/demo_examples/      6 casos de teste para a interface de demonstração (versionado)
notebooks/               EDA (TCC-01) e treino do modelo base v2/v3 (TCC-04)
scripts/                 pré-processamento, split, ISIC2020, DDI, dataset PyTorch, avaliação de
                         fairness e interface de demonstração
```
