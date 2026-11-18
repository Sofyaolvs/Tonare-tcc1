# 08 — Reprodução

Tudo o que está nesta pasta foi gerado a partir de scripts do repositório. Esta página diz o que rodar, em que
ordem, e onde cada arquivo vem.

## 1. Ambiente

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt -r requirements-train.txt   # Windows
```

`requirements-train.txt` traz `torch`, `torchvision` e `gradio`. O treino real roda no **Kaggle (GPU)**; a
avaliação roda na CPU local. Os checkpoints (`*.pt`, ~91 MB) e os dados **não vão para o git** (`.gitignore`).

## 2. Pastas

```
data/raw/<dataset>/        dados originais (gitignored)
data/processed/            imagens 224x224, manifestos split_index*.csv, checkpoints, previsões (gitignored)
data/demo_examples/        6 casos da demonstração (versionado)
notebooks/                 EDA e treino v2 a v7
scripts/                   pré-processamento, manifestos, avaliação, relatório
docs/                      esta documentação (texto, figures/, data/)
```

## 3. Preparar os dados (ordem)

| Passo | Comando / ação | Resultado |
|---|---|---|
| HAM10000 | baixar (ver README raiz) → `scripts/preprocess.py` → `scripts/split_dataset.py` | `split_index.csv` (teste idêntico em todas as versões) |
| DDI | registro manual no Stanford AIMI → `scripts/prepare_ddi.py` | `data/processed/ddi/` |
| ISIC 2020 (v3) | `scripts/prepare_isic2020.py` → `scripts/build_combined_manifest.py` | `split_index_combined.csv` |
| MILK10k (v4) | baixar `milk10k.zip` e CSVs do ISIC em `data/raw/milk10k/` → `scripts/build_milk10k_manifest.py` | `split_index_v4.csv` |
| BCN20000, HIBA, MSK 2020 (v5 e testes) | metadados via API do ISIC em `data/raw/candidates/` → `scripts/download_isic_images.py bcn20000 hiba251 consec_mel2020_216` | imagens baixadas |
| v5 | `scripts/build_v5_manifest.py` | `split_index_v5.csv` |
| v6 | `scripts/build_v6_manifest.py` | `split_index_v6.csv` (fotos clínicas ×4) |
| PAD-UFES-20 (teste) | zip do Mendeley → extrair mel/nv em `data/raw/padufes20/` | — |
| Fitzpatrick17k (v7) | zip das imagens + `fitzpatrick17k.csv` do repositório dos autores em `data/raw/fitzpatrick17k/` | — |
| v7 | `scripts/build_v7_manifest.py caminho/fitzpatrick17k.zip` | `split_index_v7.csv` e `tonare_v7_upload.zip` |

Todos os `build_*_manifest.py` **validam por código** que (a) nenhuma lesão aparece em dois splits e (b) val/teste do
HAM10000 e o conjunto reservado do MILK10k são idênticos aos das versões anteriores.

## 4. Treinar (Kaggle)

1. Subir `data/processed/images_224/` e os `split_index*.csv` como Dataset privado (a v7 também precisa do
   `tonare_v7_upload.zip`, que acrescenta as imagens `fz_*` e o manifesto).
2. Importar o notebook da versão, **anexar todos os datasets** (a Célula 5 junta as pastas `processed` de cada
   dataset anexado e confere se todas as imagens do manifesto existem antes de treinar), ligar GPU e **Run All**.
3. Baixar `model_resnet50_baseline_vN.pt` para `data/processed/`.

A v6 monta o manifesto na própria Célula 5 a partir do v5; a v7 lê o `split_index_v7.csv`.

### Mapeamento dos notebooks do Kaggle

Os notebooks **exportados do Kaggle** tinham **nomes deslocados em um** em relação às versões (o primeiro se chamava
`tonate-v1`, mas é a **v2**; a v1 original não tem notebook). Para acabar com a confusão, as **cópias com os nomes
corretos**, mantendo as saídas (logs de treino), estão em
[`notebooks/kaggle_executados/`](../notebooks/kaggle_executados/). O conteúdo de cada notebook já usava a numeração
do projeto (título e nome do checkpoint salvo); só o nome do arquivo estava errado.

| Nome original exportado | **Nome correto (neste repositório)** | Versão | Como confirmamos |
|---|---|---|---|
| `tonate-v1.ipynb` | `tonare_v2_executado.ipynb` | v2 | só HAM10000; treino 5.355; limiar 0,21 (salvava `model_resnet50_baseline.pt`, sem versão no nome) |
| `tonare-v2.ipynb` | `tonare_v3_executado.ipynb` | v3 | usa `split_index_combined.csv`; **sem saídas salvas** |
| `tonare-v4 (1).ipynb` | `tonare_v4_executado.ipynb` | v4 | treino 6.769; limiar 0,22; salva `_v4.pt` |
| `tonare-v3.ipynb` | `tonare_v5_executado.ipynb` | v5 | treino 15.954; limiar 0,14; salva `_v5.pt` |
| `tonare-v5.ipynb` | `tonare_v6_executado.ipynb` | v6 | treino 18.075; limiar 0,14; salva `_v6.pt` |
| `tonare-v6.ipynb` | `tonare_v7_executado.ipynb` | v7 | treino 19.205; limiar 0,15; salva `_v7.pt` |

A AUC de validação recalculada a partir de cada checkpoint bate com a salva nele nas 6 versões. **Daqui em
diante, nomeie o notebook no Kaggle com o número da versão do projeto** (ex.: `tonare-v7`) para não repetir o
problema. Os notebooks "limpos" do repositório (`notebooks/02_train_baseline*.ipynb`, sem saídas) usam a mesma
numeração.

## 5. Avaliar (cada versão, nos 8 testes)

```bash
python scripts/evaluate_ham_test.py v7
python scripts/evaluate_milk10k.py v7 heldout
python scripts/evaluate_padufes20.py v7
python scripts/evaluate_external.py v7 hiba251
python scripts/evaluate_external.py v7 consec_mel2020_216
python scripts/evaluate_ddi.py v7
python scripts/evaluate_fitzpatrick17k.py v7 caminho/fitzpatrick17k.zip   # na v7 exclui o que treinou
python scripts/threshold_by_modality.py v2 v4 v5 v6 v7
```

Cada script grava as previsões por imagem em `data/processed/<teste>/` (uma por versão), que alimentam o relatório.
O preprocessamento da avaliação reproduz exatamente o do treino (resize bilinear + JPEG q95).

## 6. Gerar tabelas e gráficos desta pasta

| O quê | Como | Saída |
|---|---|---|
| Comparação entre versões (tabelas + 13 gráficos) | `python scripts/build_docs_report.py` | `docs/data/*.csv`, `docs/figures/fig01…fig13` |
| Gráficos de cada versão (8 + JSON) a partir do checkpoint e dos logs dos notebooks | `python scripts/gerar_graficos_por_versao.py [pasta_dos_notebooks]` | `docs/figures/por_versao/graficos_vN/`, `docs/data/treino_historico_por_versao.csv`, `docs/data/resumo_por_versao_notebook.csv` |
| Os mesmos 8 gráficos **dentro de um notebook do Kaggle** (célula única) | colar `scripts/celula_graficos_treino.py` no fim do notebook e ajustar `NOME_VERSAO` | `/kaggle/working/graficos_<versão>.zip` |

O `build_docs_report.py` também roda no Kaggle com um pacote leve de entradas (previsões + manifestos +
`thresholds.json`, 2 MB): defina `TONARE_PROC` (pasta das entradas) e `TONARE_DOCS` (pasta de saída).

O `gerar_graficos_por_versao.py` **recalcula** as probabilidades de validação e teste e **confere** que a AUC de
validação bate com a salva no checkpoint (para se divergir). As curvas de treino vêm das linhas
`[Estágio N] época ...` dos logs dos notebooks; a v3 não tem log salvo.

## 7. Demonstração

```bash
./.venv/Scripts/python.exe scripts/frontend/demo_app.py
```

Interface local (Gradio) em `http://127.0.0.1:7860`; usa `model_resnet50_baseline_v7.pt` (versão no topo do
script). Precisa do **Python do `.venv`** (o Python global não tem o Gradio).

## 8. Tabelas em `docs/data`

| Arquivo | Conteúdo |
|---|---|
| `resultados_por_versao_e_teste.csv` | por versão × teste: nº mel/nv, AUC e IC, limiar, sensibilidade, especificidade, acurácia, acurácia balanceada, acurácia de "sempre benigno" |
| `comparacao_pareada_auc.csv` | diferenças de AUC pareadas (v3−v2, v4−v2, v5−v4, v6−v5, v7−v6, v7−v2) com IC e probabilidade |
| `fairness_por_tom_de_pele.csv` | AUC, sensibilidade e especificidade por grupo de tom (MILK10k e DDI), por versão |
| `fitzpatrick17k_tons_V_VI.csv` | AUC, sensibilidade e especificidade nos tons V–VI do Fitzpatrick17k, com a marca da v7 |
| `composicao_do_treino_por_versao.csv` | imagens por fonte e modalidade em cada treino (efetivas por época e distintas) |
| `treino_historico_por_versao.csv` | loss/AUC de treino e validação por época (extraído dos logs) |
| `resumo_por_versao_notebook.csv` | AUC val/teste, limiar, sensibilidade, especificidade, acurácia, precisão, mel perdidos, nevos marcados |
| `limiar_por_modalidade.csv` | limiares por modalidade e pontos operacionais em cada teste |
| `auditoria_datasets_isic.csv` | contagens por coleção do ISIC (API) |

## 9. Cuidados conhecidos

- Os caminhos nos manifestos foram gravados no Windows (`\`); o carregador normaliza.
- **Sempre** versionar o nome do checkpoint; uma execução já sobrescreveu outra por engano.
- Um checkpoint de **smoke test** (CPU, amostra pequena) tem `smoke_test: True` e **não** deve ser usado.
- O Kaggle só enxerga arquivos novos de um dataset depois de **reiniciar a sessão** (ou reanexar o dataset).
- Python global do Windows pode ser o atalho da Microsoft Store (`Python was not found`): use o do `.venv`.
