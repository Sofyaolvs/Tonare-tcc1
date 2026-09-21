# Sprint 1 — Preparação dos Dados (Tonare)

Data: 16/09/2026

---

## ✅ Sprint Review

**Foco: Produto / Entregas**

### 1. O que foi entregue

**Stories concluídas**

| ID | Story | Status |
|---|---|---|
| TCC-01 | Análise exploratória do dataset | ✅ Concluída |
| TCC-02 | Pré-processamento das imagens | ✅ Concluída |
| TCC-03 | Split treino/validação/teste | ✅ Concluída |
| *(ad-hoc)* | Preparação do benchmark de fairness (DDI) | ✅ Concluída |

**Demonstração funcional (demo)**

- `notebooks/01_eda.ipynb` executado de ponta a ponta sem erros: estrutura dos metadados,
  distribuição de classes, amostra visual real (4 imagens de melanoma + 4 de nevo lado a lado),
  checagem de qualidade e resumo escrito.
- `scripts/preprocess.py`, `scripts/split_dataset.py` e `scripts/prepare_ddi.py` rodados via
  CLI e validados com dados reais (não só revisão de código) — gerando os manifestos e o
  índice de split em `data/processed/`.

**Resultados alcançados**

- Pipeline de dados reprodutível (`--seed 42` fixo) e determinístico, do download bruto até o
  manifesto pronto para treino.
- **13.188 imagens** de treino (mel vs nv, HAM10000) após limpeza e augmentation balanceado.
- **656 imagens do DDI** processadas com 100% de aproveitamento (0 arquivos faltando),
  números batendo exatamente com o paper original (208/241/207 por faixa de tom de pele).
- Split 70/15/15 validado programaticamente: **zero vazamento de lesão** entre treino/val/teste.
- Pivô de escopo de dataset (ver Retrospectiva) bem fundamentado e documentado, evitando que
  o TCC seguisse em cima de uma base de dados genérica demais para o objetivo real.

### 2. O que não foi concluído

| Item | Motivo |
|---|---|
| Classificação nas 7 classes completas do HAM10000 | Mudança de escopo — decisão consciente de focar em `mel` vs `nv` agora e adiar as 7 classes para uma sprint futura |
| Treino do modelo classificador | Fora do escopo desta sprint (é a Sprint 2) |
| Avaliação de fairness do modelo no DDI | Depende do modelo treinado existir (Sprint 2) — a *preparação* dos dados do DDI já está pronta |
| Resultado de fairness específico de melanoma (não só maligno/benigno) | Limitação de dado: o DDI só tem 21 casos de melanoma no total (~7 por faixa de tom de pele) — insuficiente para uma métrica robusta por subgrupo; registrado como limitação, não resolvido |
| Investigação manual dos 278 pares de imagens quase-idênticas entre lesões diferentes (HAM10000) | Identificado na EDA, mas não investigado a fundo — pode indicar erro de rotulagem no dataset original |

### 3. Métricas relevantes

- **Volume de dados:** 7.818 imagens mel+nv originais → 7.678 após limpeza de qualidade →
  13.188 após augmentation (mel balanceado de 1.113 para ~6.612).
- **Qualidade dos dados (HAM10000):** 0 imagens corrompidas, 2 duplicatas exatas, 278 pares
  quase-idênticos entre lesões diferentes (flag para investigação).
- **Split:** 70,0% / 15,1% / 15,1% (treino/val/teste), proporção de classe preservada em todos
  (~50/50 após augmentation); validação de vazamento de lesão: **0 ocorrências**.
- **DDI:** 656/656 imagens processadas sem falha; cruzamento malignidade × tom de pele com
  volume estatisticamente usável (49–74 casos malignos por faixa).
- **Bugs encontrados e corrigidos durante a sprint:**
  - Mojibake de acentuação em outputs de notebook/scripts no Windows (corrigido forçando
    `PYTHONIOENCODING=utf-8`).
  - Processo de download duplicado ao rodar em background via `nohup` no git-bash do Windows
    (corrigido usando o controle de background nativo da ferramenta em vez de `nohup &`).
- **Velocidade:** sprint concluída em uma única sessão de trabalho, incluindo um pivô completo
  de dataset no meio do caminho (retrabalho de TCC-01/02/03 refeito do zero em cima do novo dataset).

### 4. Feedback do cliente/stakeholders

- **Ajuste solicitado (pela própria autora, no meio da sprint):** o dataset inicial
  (Fitzpatrick17k) cobria 114 condições de pele genéricas — o objetivo real do TCC é
  **melanoma especificamente**, não dermatologia geral. Pipeline inteiro foi refeito para
  refletir isso.
- **Nova prioridade definida:** já que nenhum dataset público único tem volume de melanoma E
  diversidade de tom de pele ao mesmo tempo, a fairness passou a ser avaliada num benchmark
  separado e dedicado (DDI), em vez de tentar espremer as duas coisas de um dataset só.

📌 **Resumo:** a sprint entregou um pipeline de dados funcional e validado para treino de um
classificador de melanoma (mel vs nv) mais um benchmark de fairness por tom de pele pronto —
mas passou por uma mudança de escopo relevante no meio do caminho.

---

## ✅ Sprint Retrospective

**Foco: Processo / Time**

### 1. O que funcionou bem

- **Validação com dados reais em cada etapa**, não só revisão de código — cada script foi
  rodado de fato contra os dados baixados antes de ser considerado pronto, o que pegou
  problemas (ex: encoding, processos duplicados) antes de virarem entregas quebradas.
- **Descoberta cedo de bloqueio externo:** o domínio `dermaamin.com` fora do ar foi detectado
  e confirmado (via DNS público, não achismo) antes de a sprint inteira depender dele.
- **Arquitetura reutilizável:** funções de qualidade (corrompidas/duplicatas), normalização e
  augmentation generalizaram bem quando o dataset trocou inteiro (Fitzpatrick17k → HAM10000).
- **Decisões de escopo levadas para validação** em vez de resolvidas silenciosamente (ex: 3
  classes vs. binário, `mel` vs `nv` vs 7 classes, qual dataset usar) — evitou retrabalho maior
  mais à frente.

### 2. O que pode melhorar

- **Validar a disponibilidade real de um dataset ANTES de montar o pipeline completo em cima
  dele.** O problema do `dermaamin.com` só apareceu depois que boa parte do TCC-01/02/03 já
  tinha sido implementada para o Fitzpatrick17k — um teste rápido de amostra logo no início
  teria evitado esse retrabalho.
- **Falta de refinamento prévio do enquadramento clínico.** A escolha de dataset e de classes
  (melanoma vs. lesão de pele em geral) não foi validada com a orientação antes de começar a
  implementar — o pivô no meio da sprint custou tempo que um alinhamento prévio teria evitado.
- **Ambiente Windows/git-bash** gerou instabilidades pontuais (encoding de console, processos
  em background duplicados) que não aconteceriam num ambiente Linux/Mac — vale considerar
  documentar esses gotchas para não perder tempo de novo.

### 3. Plano de ação

| Ação | Responsável | Prazo |
|---|---|---|
| Validar disponibilidade real de qualquer novo dataset (baixar uma amostra pequena primeiro) antes de construir pipeline completo | Claude / Sofya | Início de cada sprint que envolva dado novo |
| Confirmar enquadramento clínico (classes, dataset, escopo) com a orientação antes de iniciar a Sprint 2 (treino do modelo) | Sofya | Antes do início da Sprint 2 |
| Expandir o classificador para as 7 classes completas do HAM10000 | A definir | Sprint futura (pós-Sprint 2) |
| Investigar manualmente os 278 pares de imagens quase-idênticas entre lesões diferentes flagados na EDA | Sofya | Antes de finalizar o treino do modelo |
| Documentar os gotchas de ambiente Windows (encoding, background de processos) no README do projeto | Claude | Próxima sessão de trabalho |
