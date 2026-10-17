# Tonare — TCC

Triagem de **melanoma** por imagem (CNN + transfer learning) com análise de equidade
(fairness) entre tons de pele.

> **Documentação completa em [`docs/`](docs/README.md):** metodologia, datasets (incluindo a auditoria do arquivo do
> ISIC e os descartados), histórico das versões v1 a v7, comparação com IC 95% e testes pareados, equidade por tom de
> pele, limiares e métricas, limitações e como reproduzir — com gráficos e tabelas em CSV.
> **Modelo de referência atual: v7** (dermatoscopia: AUC 0,923, sensibilidade 0,861, especificidade 0,796).
> Este README é o guia rápido; os detalhes e o histórico completo estão em `docs/`.

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

- **v4** ([`notebooks/02_train_baseline_v4_milk10k.ipynb`](notebooks/02_train_baseline_v4_milk10k.ipynb)):
  soma o **MILK10k** (foto clínica + dermatoscopia das mesmas lesões) ao treino, para atacar a troca
  de modalidade encontrada na Sprint 3. Manifesto em
  [`scripts/build_milk10k_manifest.py`](scripts/build_milk10k_manifest.py): split por lesão, 30% das
  lesões reservadas (`test_milk`) para comparar versões em dados inéditos, teste do HAM10000 idêntico
  ao da v2, lesões do MILK10k parecidas com HAM10000/ISIC2020 descartadas por phash.
  Melhorou a foto clínica (AUC 0,70 → 0,79 no MILK10k reservado).
- **v5** ([`notebooks/02_train_baseline_v5_bcn20000.ipynb`](notebooks/02_train_baseline_v5_bcn20000.ipynb)):
  v4 + **BCN20000** (Hospital Clínic de Barcelona, 10.283 mel/nv de dermatoscopia, ~920 lesões de
  melanoma; [`scripts/build_v5_manifest.py`](scripts/build_v5_manifest.py), 8 lesões descartadas por
  phash). Melhorou a dermatoscopia em população real (MSK 2020, AUC 0,67 → 0,76), mas as fotos
  clínicas caíram para ~4% do treino e o ganho clínico da v4 se perdeu (0,79 → 0,72).
- **v6** ([`notebooks/02_train_baseline_v6_clinical_oversample.ipynb`](notebooks/02_train_baseline_v6_clinical_oversample.ipynb)):
  mesmos dados da v5, com as fotos clínicas do MILK10k repetidas 4x no treino (4% → 16%,
  [`scripts/build_v6_manifest.py`](scripts/build_v6_manifest.py)); o notebook também escolhe limiares
  por modalidade na validação. Mantém o ganho clínico da v4 e o da v5 em dermatoscopia real
  (tabela abaixo).

- **v7** ([`notebooks/02_train_baseline_v7_fitzpatrick.ipynb`](notebooks/02_train_baseline_v7_fitzpatrick.ipynb)):
  v6 + 629 fotos clínicas do [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) **só dos tons I-IV**
  (421 melanomas / 208 nevos, repetidas 2x; [`scripts/build_v7_manifest.py`](scripts/build_v7_manifest.py)).
  Tons V-VI, tom sem anotação e "lentigo maligna" (todas de outra fonte) ficam fora do treino. Rótulos de
  atlas (sem biópsia garantida) e sem id de lesão: as imagens foram agrupadas por phash para o split.
  **Modelo de referência atual**, por pouco: nas fontes independentes a v7 é igual ou ligeiramente melhor
  que a v6 (teste pareado abaixo); só o HAM10000 tem melhora estatisticamente sustentada (+0,015).

### Como rodar no Kaggle

1. O Kaggle já vem com PyTorch+GPU (`requirements-train.txt` documenta as versões usadas local).
2. Suba `data/processed/` (pasta `images_224/` + `preprocessed_manifest.csv` + `split_index.csv`
   + `split_index_combined.csv`, se for rodar a v3) como um Kaggle Dataset privado.
3. Crie um notebook, anexe o dataset, ative GPU e Internet (exige celular verificado na conta).
4. Copie as células do notebook (v2 a v7); na Célula 1, no lugar do import, cole o conteúdo de
   `scripts/torch_dataset.py` (import relativo não funciona lá) e ajuste o caminho
   `KAGGLE_PROCESSED` na célula de dados para o seu dataset.

Checkpoints resultantes (baixados do Kaggle, não vão pro git — ver `.gitignore`):
`data/processed/model_resnet50_baseline_v7.pt` (referência atual) e as versões anteriores `_v2.pt` a `_v6.pt`.
Nomes sempre com a versão no arquivo, de propósito: um checkpoint sem sufixo já foi sobrescrito por engano.

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
não resolve a troca de modalidade. Por isso, na época, o v2 foi o modelo de referência, não o v3 (a v7 é a referência atual — ver abaixo).

### Extensão: avaliação no MILK10k (dermatoscopia vs foto clínica, por tom de pele)

[`scripts/evaluate_milk10k.py`](scripts/evaluate_milk10k.py) usa o [MILK10k](https://api.isic-archive.com/doi/milk10k/)
(ISIC, CC-BY-NC): cada lesão tem **foto clínica e dermatoscopia**, o que separa o efeito do tipo de
imagem do efeito do tom de pele (misturados no DDI). Só teste, nada entra no treino. Entram as
1.196 lesões mel/nv (450 melanomas); tom de pele na escala própria do MILK10k (0 = muito escuro,
5 = muito claro, não é Fitzpatrick); IC95% por bootstrap de lesões. Download manual dos CSVs/zip em
`data/raw/milk10k/` (links na página do dataset).

| v2, limiar 0,21 | AUC | Sensibilidade | Especificidade |
|---|---|---|---|
| Dermatoscopia (mesmo tipo do treino) | 0,847 | 0,876 | 0,621 |
| Foto clínica | 0,660 | 0,340 | 0,875 |

- **Dermatoscopia:** sem diferença detectável entre tons (AUC 0,84 escuro / 0,84 médio / 0,87 claro; ICs sobrepostos).
- **Foto clínica:** a AUC cai ~0,19 e o limiar (escolhido em dermatoscopia) deixa passar 66% dos melanomas, em todos os tons — a queda é de **modalidade**, como suspeitado no DDI.
- **Ressalva de cobertura:** só 16 melanomas nos tons 0-1; "escuro" aqui é sobretudo o nível 2. Fitzpatrick V-VI praticamente não existe neste dataset.

### Comparação de versões em testes externos (nenhum entra no treino)

Testes: HAM10000 (teste original), 30% das lesões do MILK10k reservadas, PAD-UFES-20 (foto de celular,
Brasil, 52 imagens de melanoma), HIBA (Buenos Aires, dermatoscopia + foto clínica), MSK 2020 (biópsias
consecutivas, população de encaminhamento real) e DDI. IC95% por bootstrap de lesão nos CSVs/scripts.
A v7 usou o Fitzpatrick17k (tons I-IV) no treino; as demais fontes nunca entram no treino de nenhuma versão.

| AUC | v2 | v4 | v5 | v6 | **v7** |
|---|---|---|---|---|---|
| HAM10000, dermatoscopia | 0,922 | 0,922 | 0,916 | 0,908 | 0,923 |
| MILK10k reservado, foto clínica | 0,696 | 0,790 | 0,717 | 0,784 | 0,774 |
| MILK10k reservado, dermatoscopia | 0,865 | 0,884 | 0,859 | 0,862 | 0,850 |
| PAD-UFES-20, foto de celular | 0,642 | 0,722 | 0,755 | 0,807 | 0,823 |
| HIBA, dermatoscopia | 0,895 | 0,895 | 0,912 | 0,918 | 0,925 |
| HIBA, foto clínica | 0,774 | 0,764 | 0,739 | 0,753 | 0,799 |
| MSK 2020, população real | 0,653 | 0,665 | 0,764 | 0,790 | 0,791 |
| DDI (maligno vs benigno) | ~0,62 | 0,662 | 0,667 | 0,660 | 0,681 |

- **O ganho da v6 sobre a v2 é claro** (ex.: MSK 0,65 → 0,79, ICs separados). Entre v5 e v6 as diferenças
  estão, em vários testes, dentro do intervalo de confiança. A AUC no HAM10000 tem tendência de queda
  (0,922 → 0,908), dentro do ruído; a sensibilidade no limiar 0,14 do HAM10000 é 0,78 (v2: 0,885).
- **v7 x v6, teste pareado** (mesmas imagens, bootstrap de lesão; diferença de AUC v7 − v6): HAM10000 +0,015
  [+0,001 a +0,030]; HIBA clínica +0,046; DDI +0,021; PAD-UFES-20 +0,017; HIBA dermatoscopia +0,007; MSK
  +0,001; MILK10k clínica −0,010; MILK10k dermatoscopia −0,013 — todas, exceto o HAM10000, com intervalo
  que inclui o zero. No limiar 0,15 o HAM10000 dá sensibilidade 0,86 e especificidade 0,80.
- **CUIDADO com o Fitzpatrick17k:** nos tons V-VI (43 melanomas / 24 nevos, nunca vistos no treino) a AUC
  foi de 0,67 (v6) para 0,93 (v7), mas nos tons I-IV de validação (mesma fonte, também nunca treinados)
  foi de 0,59 para 0,87. É ganho **na própria fonte** (estilo do atlas), não prova de generalização para
  pele escura: o desenho "treina I-IV, testa V-VI" não é independente porque a fonte é a mesma. Só dá para
  dizer que, dentro do atlas, V-VI (0,93) ficou parecido com I-IV (0,87).
- **Foto clínica continua sendo o ponto fraco** (AUC 0,75-0,82). No limiar único 0,14, a v6 dá
  sensibilidade 0,78-0,88 com especificidade 0,35-0,64 em foto clínica; v4/v5 chegavam a ~0,3 de
  especificidade. Resultados por limiar e política em `data/processed/threshold_by_modality.csv`
  ([`scripts/threshold_by_modality.py`](scripts/threshold_by_modality.py)): nenhuma regra de limiar dá
  sensibilidade e especificidade altas ao mesmo tempo em foto clínica — falta dado clínico.
- **Tom de pele:** sem diferença sustentada entre tons no MILK10k nem no DDI em nenhuma versão, mas com
  poucos casos nas faixas escuras. Melanoma em Fitzpatrick V-VI praticamente não existe em nenhum
  dataset aberto auditado (MILK10k, HIBA, PAD-UFES-20, DDI tem 7 por faixa) — é limitação do TCC.
- **Meta de "98% de acurácia" não é um objetivo adequado:** com ~86% de nevos no teste, prever sempre
  "benigno" já dá ~86% de acurácia; resultados ≥98% na literatura costumam vir de vazamento entre
  treino e teste. Usamos AUC e sensibilidade/especificidade.
- **Datasets descartados após auditoria (API do ISIC):** coleção "Mel+Nev histopatologia" (294), que repete
  imagens do HAM10000/ISIC2020/MILK10k sob outros IDs; ISIC 2024 (401 mil recortes, só 157 melanomas);
  DERM12345 (só 400 melanomas); SCIN, BALD. O Fitzpatrick17k, descartado na primeira tentativa por
  links mortos, foi recuperado a partir do zip das imagens + CSV dos autores e entrou na v7 (ver acima).

```bash
python scripts/evaluate_ham_test.py v7
python scripts/evaluate_milk10k.py v7 heldout
python scripts/evaluate_padufes20.py v7
python scripts/evaluate_external.py v7 hiba251          # imagens: scripts/download_isic_images.py hiba251
python scripts/evaluate_external.py v7 consec_mel2020_216
python scripts/evaluate_ddi.py v7
python scripts/evaluate_fitzpatrick17k.py v7 caminho/fitzpatrick17k.zip   # na v7 exclui o que treinou
python scripts/threshold_by_modality.py v6 v7
```

```bash
python scripts/prepare_isic2020.py       # extrai/dedupe o ISIC2020 contra o HAM10000 (opcional, só p/ v3)
python scripts/build_combined_manifest.py  # gera split_index_combined.csv (opcional, só p/ v3)
python scripts/evaluate_ddi.py          # v2 (padrão); aceita a versão: evaluate_ddi.py v6
python scripts/analyze_ddi_by_disease.py
python scripts/evaluate_milk10k.py       # requer data/raw/milk10k/ (ver acima)
```

## Interface de demonstração

```bash
python scripts/demo_app.py
```
Abre em `http://127.0.0.1:7860`. Interface local (Gradio) para testar o classificador com uma
foto solta na tela ou um dos 6 casos de exemplo em `data/demo_examples/`. Estilo em
[`scripts/tonare.css`](scripts/tonare.css).
Usa `data/processed/model_resnet50_baseline_v7.pt` (versão em `CHECKPOINT_VERSION`, no topo do script; ver Sprint 2).
Os 6 exemplos são do teste do HAM10000 e foram escolhidos com a v2: com a v7, 5 de 6 estão corretos
(os 3 melanomas, casos 1, 3 e 5, são marcados como suspeitos; o nevo do caso 4 vira falso positivo com 21,6%).
São 6 imagens ilustrativas, sem valor estatístico.

## Estrutura

```
data/raw/ham10000/       metadados + imagens originais do HAM10000 (gitignored)
data/raw/ddi/            metadados + imagens do DDI, após você baixar manualmente (gitignored)
data/raw/isic2020/       imagens mel/nv extraídas do ISIC2020, dedupe contra o HAM10000 (gitignored)
data/raw/milk10k/        MILK10k: zip + CSVs do ISIC (gitignored)
data/raw/padufes20/      PAD-UFES-20: metadados + imagens mel/nv (gitignored)
data/raw/candidates/     metadados e imagens do ISIC para BCN20000, HIBA e MSK 2020 (gitignored)
data/raw/fitzpatrick17k/ CSV dos autores + imagens extraídas do zip (gitignored)
data/processed/          imagens redimensionadas, manifestos, índices de split e checkpoints (gitignored)
data/demo_examples/      6 casos de teste para a interface de demonstração (versionado)
notebooks/               EDA (TCC-01) e treino dos modelos v2 a v7 (TCC-04)
scripts/                 pré-processamento, split, manifestos v3 a v7, avaliação nos testes externos,
                         relatório/gráficos da documentação e interface de demonstração
docs/                    documentação da pesquisa (texto, figures/ e data/): ver docs/README.md
```
