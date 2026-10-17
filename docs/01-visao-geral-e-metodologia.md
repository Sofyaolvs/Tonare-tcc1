# 01 — Visão geral e metodologia

## 1. Problema

O melanoma é o câncer de pele mais letal e, detectado cedo, tem boa chance de cura; o acesso rápido a um
dermatologista, porém, é desigual. Modelos de IA para lesões de pele costumam ser treinados quase só com pele
clara, então podem ir bem na média e mal justamente em quem já tem menos acesso a diagnóstico.

O Tonare tem dois objetivos:

1. **Classificar** lesões como melanoma (`mel`) ou nevo benigno (`nv`) a partir de uma imagem;
2. **Medir** se o desempenho muda entre tons de pele — e, quando os dados não permitem concluir, dizer isso.

Escopo deliberado: triagem binária `mel` vs `nv` (as 7 classes do HAM10000 ficam como trabalho futuro).

## 2. Visão geral do pipeline

```
Datasets públicos ──► pré-processamento (224x224) ──► split por LESÃO ──► treino (Kaggle, GPU)
                                                                              │
         testes externos (nunca no treino) ◄── checkpoint + limiar (só na validação)
                 │
                 └─► AUC / sensibilidade / especificidade com IC 95% ──► comparação entre versões
```

### Sprint 1 — dados
- EDA (`notebooks/01_eda.ipynb`), pré-processamento (`scripts/preprocess.py`) e split (`scripts/split_dataset.py`).
- Imagens redimensionadas para **224×224 com interpolação bilinear do PIL e gravadas em JPEG qualidade 95**.
  Esse detalhe importa muito (ver §6).
- Limpeza: imagens corrompidas, duplicatas exatas e quase-duplicatas (phash).
- **Split 70/15/15 estratificado por classe e agrupado por LESÃO**, não por imagem: no HAM10000 uma mesma lesão
  aparece em várias fotos, e separar por imagem faria o modelo "ver" no treino a lesão que será testada.

### Sprint 2 — treino
- **Modelo:** ResNet50 pré-treinado no ImageNet (pesos `IMAGENET1K_V2`), camada final de 2 saídas.
- **Treino em dois estágios:**
  1. só a camada final (backbone congelado), AdamW, LR 1e-3, weight decay 1e-2, até 5 épocas;
  2. rede inteira (fine-tuning), AdamW, LR 1e-5, weight decay 1e-2, até 12 épocas.
- **Loss:** entropia cruzada com pesos de classe "balanced" calculados do treino de cada versão (o peso do melanoma
  cai de 3,47 na v2 para ~1,4 nas versões com BCN20000, porque o treino fica menos desbalanceado).
- **Augmentation aleatório a cada época** (rotação livre, recorte, flips, brilho/contraste leves), **sem alterar
  matiz/saturação** — a cor é sinal diagnóstico e também o atributo que a análise de equidade observa.
- **Early stopping** pela AUC de validação (paciência 3) e **restauração da melhor época**.
- Batch 32, semente 42. Treino real no **Kaggle (GPU)**; esta máquina não tem GPU dedicada e só faz avaliação.
- **Limiar de decisão:** o **maior** limiar cuja sensibilidade na **validação** seja ≥ 0,90 (maior limiar = menos
  falsos positivos dentro da meta). O teste nunca participa dessa escolha.

### Sprint 3 — avaliação e equidade
- Avaliação em **fontes externas independentes** (ver [02](02-datasets.md)): MILK10k reservado, PAD-UFES-20, HIBA,
  MSK 2020 e DDI, além do teste do HAM10000.
- Análise por tom de pele ([05](05-fairness-tom-de-pele.md)) e análise de limiar ([06](06-limiares-e-metricas.md)).

## 3. Como as versões foram comparadas

Cada versão muda **uma coisa** em relação à anterior (ver [03](03-historico-v1-a-v7.md)), com o mesmo teste do
HAM10000 e as mesmas lesões reservadas do MILK10k, para que qualquer diferença seja atribuível à mudança.

Todas as versões são avaliadas nos **mesmos 8 testes**:

| Teste | Tipo de imagem | Para que serve |
|---|---|---|
| HAM10000 (teste original) | dermatoscopia | desempenho "em casa" |
| MILK10k reservado, dermatoscopia | dermatoscopia | mesma lesão que a foto clínica abaixo |
| MILK10k reservado, foto clínica | foto clínica | efeito da troca de modalidade |
| PAD-UFES-20 | foto de celular (Brasil) | uso real, foto comum |
| HIBA, dermatoscopia | dermatoscopia (Buenos Aires) | outra instituição |
| HIBA, foto clínica | foto clínica | outra instituição, foto |
| MSK 2020, biópsias consecutivas | dermatoscopia | população real de encaminhamento |
| DDI | foto clínica | tons de pele diversos, biópsia |

## 4. Estatística

- **Intervalos de confiança de 95% por bootstrap de lesão** (2.000 reamostragens nos scripts de avaliação, 1.000 no
  relatório): as fotos da mesma lesão são reamostradas juntas.
- **Comparação entre versões:** bootstrap **pareado** (as mesmas lesões reamostradas para as duas versões), que
  tem mais poder que comparar intervalos a olho. A coluna "exclui zero?" indica diferença sustentada.
- **Comparações múltiplas:** são 8 testes; um resultado isolado com p ≈ 0,01 deve ser lido com cautela. O que damos
  como "sustentado" é o que aparece de forma consistente em mais de um teste.
- Poucos casos (menos de 10 em alguma classe) são **descritivos**, sem métrica robusta.

## 5. Controles contra vazamento de dados

| Controle | Onde |
|---|---|
| Split por **lesão**, nunca por imagem | todos os datasets com id de lesão |
| Teste do HAM10000 **idêntico** em todas as versões (validado por código) | `build_*_manifest.py` |
| 30% das lesões do MILK10k **reservadas** (`test_milk`), fora de qualquer treino | `build_milk10k_manifest.py` |
| Imagens quase idênticas (phash) a imagens já usadas são **descartadas** | MILK10k: 17 lesões; BCN20000: 8 lesões; Fitzpatrick17k: 2 imagens |
| Sobreposição exata de ID/lesão com treino/teste checada via API do ISIC | auditoria ([02](02-datasets.md)) |
| Coleção do ISIC com **imagens duplicadas** sob outros IDs **não usada** | coleção 294 ([02](02-datasets.md)) |
| Limiar escolhido **só na validação** | notebooks |
| Testes externos **nunca** no treino | todas as versões (exceção: Fitzpatrick17k na v7, ver [05](05-fairness-tom-de-pele.md)) |

## 6. Armadilhas técnicas encontradas (e como foram resolvidas)

- **Pré-processamento idêntico ao do treino.** `torchvision.transforms.Resize` aplica antialiasing por padrão e
  produz uma imagem sutilmente diferente do `resize` bilinear puro do PIL usado no treino: isso mudou a
  classificação de casos limítrofes (probabilidade 0,67 contra 0,21 no mesmo caso). Solução: reproduzir o
  resize do PIL à mão.
- **Recompressão JPEG.** Mesmo com o resize idêntico, a recompressão (qualidade 95) mudou ~1,5/255 de intensidade
  média por pixel e moveu a saída em até 45 pontos percentuais em casos limítrofes. Por isso as imagens de
  teste e da demo passam pelo **mesmo pipeline resize + JPEG q95** do treino.
- **Caminhos gravados no Windows** (`\`) quebravam no Kaggle (Linux); normalizados no carregador.
- **Checkpoint sobrescrito por engano** (uma corrida apagou a outra). Desde então os arquivos têm a versão no
  nome (`model_resnet50_baseline_v7.pt`) e os scripts recebem a versão explicitamente.
- **Verificação do mapeamento dos notebooks:** a AUC de validação recalculada a partir de cada checkpoint bate
  exatamente com a salva nele (6 de 6 versões), o que confirma que cada arquivo é o que se espera.

## 7. Métricas (resumo; detalhes em [06](06-limiares-e-metricas.md))

- **AUC-ROC:** chance de o modelo dar probabilidade maior a um melanoma do que a um nevo escolhidos ao acaso;
  não depende de limiar. 0,5 = sorte, 1,0 = perfeito.
- **Sensibilidade:** dos melanomas de verdade, quantos o modelo marca como suspeitos.
- **Especificidade:** dos nevos benignos, quantos o modelo deixa em paz.
- **Acurácia** é reportada, mas **não** é a métrica principal: com classes desbalanceadas ela engana.
