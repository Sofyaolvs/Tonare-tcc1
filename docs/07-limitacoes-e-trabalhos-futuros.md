# 07 — Limitações e trabalhos futuros

## 1. Limitações (o que **não** conseguimos afirmar)

### Dados
1. **Melanoma em pele escura.** É a principal limitação. Nos datasets abertos auditados há, no máximo, algumas
   dezenas de melanomas em tons V–VI (43 no Fitzpatrick17k, 7 no DDI; **0** no arquivo inteiro do ISIC com esse tom
   anotado), e os tons realmente escuros do MILK10k (níveis 0–1) têm 16 ([02](02-datasets.md#5-cobertura-de-tons-de-pele-o-gargalo-do-projeto)).
   **Não conseguimos afirmar nem descartar viés por tom de pele** ([05](05-fairness-tom-de-pele.md)).
2. **Foto clínica.** O modelo foi treinado majoritariamente com dermatoscopia; mesmo com fotos clínicas no treino
   (v4, v6, v7), a AUC em foto clínica fica em 0,75–0,82 e a especificidade é baixa.
3. **Qualidade dos rótulos.** Fitzpatrick17k vem de atlas online, sem biópsia garantida, com tom ruidoso (duas
   anotações concordam em 45,8%). No PAD-UFES-20 só 25% dos nevos têm biópsia. O DDI rotula "maligno vs benigno" de 78
   doenças, não "melanoma vs nevo".
4. **Viés de seleção.** Os nevos biopsiados (MSK 2020) são os atípicos, e HAM10000/MILK10k/BCN20000 vêm de serviços
   especializados: não representam uma população de triagem. A prevalência de melanoma nos testes (de ~14% a ~53%) é muito
   maior que a de uma triagem real, então **valores preditivos reais seriam bem piores**.
5. **Recorte da tarefa.** Classificação binária `mel` vs `nv`; outras lesões (carcinomas, ceratoses, linfoma) não
   fazem parte do treino e o modelo as força para uma das duas classes.

### Modelo e treino
6. **Uma única rodada por versão** (semente 42), uma arquitetura (ResNet50), 224×224. **Não medimos a variação entre
   sementes**, então diferenças pequenas (da ordem de 0,01–0,02 de AUC, como v7 × v6) podem estar dentro da
   variabilidade de treino. Não há busca de hiperparâmetros, conjunto de modelos nem TTA.
7. **Limiar único mistura modalidades** na validação; a validação clínica é pequena (44 melanomas e 74 nevos até a
   v6, 87 e 95 na v7), então os limiares por modalidade são ruidosos ([06](06-limiares-e-metricas.md)).
8. **Imagens esticadas para 224×224 sem recorte**, o que distorce lesões em fotos não quadradas.

### Avaliação
9. O **teste do HAM10000 é o mesmo em todas as versões**: decisões repetidas sobre ele (escolha de versão) podem
   superajustar a ele; mitigamos com os testes externos, que não participam de nenhuma escolha de treino.
10. **Sem correção para comparações múltiplas** (8 testes × várias comparações) — só se chama de "sustentado" o que
    se repete em mais de um teste.
11. O MILK10k reservado vem da mesma fonte do que as versões v4–v7 viram no treino, então ele **favorece** essas
    versões; PAD-UFES-20, HIBA e DDI (fontes independentes) pesam mais ([04 §9](04-resultados-comparacao.md#9-ressalvas-estatísticas)).
12. O **Fitzpatrick17k na v7 não é teste independente** ([05](05-fairness-tom-de-pele.md#42-o-que-deu-errado-na-v7)).
13. A **v1** não é reprodutível (sem checkpoint nem notebook).

### Uso
14. **Não é dispositivo médico, não foi validado clinicamente e não houve estudo prospectivo.** A demonstração
    (`scripts/frontend/demo_app.py`) diz isso na própria interface.
15. **Licenças:** os termos do ISIC 2020 e dos atlas do Fitzpatrick17k não foram verificados; os demais são de uso
    acadêmico/não comercial.

## 2. Por que 98% de acurácia não é um objetivo adequado

([06](06-limiares-e-metricas.md#4-acurácia-e-a-meta-de-98) detalha.) Com ~86% de nevos no teste, "sempre benigno" já dá
85,8%; o máximo **válido** da v7 no HAM10000 é ~91%, e só perdendo metade dos melanomas. Resultados ≥ 98% na
literatura costumam vir de vazamento entre treino e teste (split por imagem em vez de por lesão, duplicatas).

## 3. Trabalhos futuros (do mais barato ao mais caro)

| # | Ideia | Esforço | Por quê / o que esperar |
|---|---|---|---|
| 1 | **Conjunto de modelos (v4 a v7) + TTA** (média das probabilidades de imagens giradas/espelhadas) | sem treino | pequeno ganho de AUC (a verificar); barato, usa o que já existe |
| 2 | **Fusão das duas fotos da mesma lesão** (clínica + dermatoscopia) | sem treino | o MILK10k tem os pares; na prática clínica costumam existir os dois |
| 3 | **Faixas de risco** (baixo/médio/alto, dois limiares) em vez de sim/não; casos incertos vão para avaliação | baixo | melhora sensibilidade/especificidade nos casos decididos |
| 4 | **Estimar a variação entre sementes** (3–5 rodadas da v7) | médio (GPU) | dá o ruído real para comparar versões |
| 5 | **Resolução maior e outro backbone** (EfficientNet/ConvNeXt, 384 px) | médio (GPU) | dermatoscopia costuma ganhar com mais resolução |
| 6 | **Metadados** (idade, sexo, local da lesão — existem no HAM10000, MILK10k, BCN20000, HIBA) | médio | costuma ajudar em melanoma; risco de atalho |
| 7 | **Normalização de cor e recorte da lesão** para foto clínica | médio | foto clínica tem fundo e iluminação variáveis |
| 8 | **Validação clínica de limiares por modalidade** com validação clínica maior | médio | hoje o limiar clínico é ruidoso |
| 9 | **Medir especificidade por tom** com o MSKCC Skin Tone Labeling (4.879 imagens benignas, I–VI balanceado) | baixo | mede falsos positivos por tom em dermatoscopia |
| 10 | **Anotar tom de pele automaticamente** nos datasets (ferramenta do [arXiv 2602.10265](https://www.arxiv.org/abs/2602.10265), se publicada) | médio | permite estratificar por tom em HAM10000/ISIC |
| 11 | **Mais melanoma em foto clínica com biópsia, incluindo pele escura**, via **parceria com serviço de dermatologia** (CEP/Plataforma Brasil, consentimento, LGPD) | alto, meses | é o gargalo real e não existe dado aberto; depende do orientador e de aprovação ética |
| 12 | **Estender para as 7 classes do HAM10000** | médio | previsto desde o início |

Nenhuma dessas ideias leva a 98% de acurácia; um teto honesto em dermatoscopia fica na faixa de AUC 0,93–0,95 (estimativa),
e foto clínica continua menor.

## 4. Notas éticas

- Pesquisa acadêmica com dados públicos já anonimizados; nenhum dado de paciente foi coletado.
- Ferramenta de triagem/apoio à decisão, **nunca** de diagnóstico; não substitui avaliação médica.
- Dados de saúde são dados pessoais sensíveis (LGPD); qualquer coleta futura exige aprovação em comitê de ética e
  consentimento.
- Não divulgar a AUC da v7 nos tons V–VI do Fitzpatrick17k como desempenho em pele escura ([05](05-fairness-tom-de-pele.md)).
