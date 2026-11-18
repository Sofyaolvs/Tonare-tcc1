"""
Interface de demonstração do Tonare — plataforma de APOIO À DECISÃO CLÍNICA (uso por profissionais de saúde).

Mantém a identidade visual do design (paleta, tipografia, navegação, hero e cartões; ver
scripts/frontend/tonare.css), com conteúdo voltado ao profissional: faixa de risco em vez de "sim ou não",
aviso do tipo de imagem, estatísticas da faixa no teste interno, desempenho e limitações do modelo.

Uso:
    python scripts/frontend/demo_app.py
Depois abre http://127.0.0.1:7860 no navegador (o Gradio também tenta abrir sozinho).
"""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_VERSION = "v7"
CHECKPOINT_PATH = BASE_DIR / "data" / "processed" / f"model_resnet50_baseline_{CHECKPOINT_VERSION}.pt"
EXAMPLES_DIR = BASE_DIR / "data" / "demo_examples"
CSS_PATH = Path(__file__).resolve().parent / "tonare.css"
HERO_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "hero_pele.jpg"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# NAO usar transforms.Resize aqui: ele aplica antialiasing por padrao e produz uma imagem
# sutilmente diferente do resize usado no treino (scripts/preprocess.py fez
# Image.resize(IMG_SIZE, Image.BILINEAR) puro do PIL, sem esse antialiasing). Essa diferenca
# já mudou a classificação de casos limítrofes num teste manual - reproduzir o resize do PIL
# aqui, à mão, e só depois converter pra tensor.
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess(img: PILImage.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((224, 224), PILImage.BILINEAR)
    return _TO_TENSOR(img)


def load_model() -> tuple[torch.nn.Module, float]:
    ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, float(ck["threshold"])


MODEL, THRESHOLD = load_model()
print(f"Modelo carregado ({CHECKPOINT_VERSION}). Limiar da meta de sensibilidade 0,90 (escolhido na validação): {THRESHOLD:.2f}")


def predict(image: PILImage.Image) -> float:
    """predict(img) -> pontuação do modelo para melanoma (0 a 1). NÃO é uma probabilidade calibrada."""
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)
        return F.softmax(MODEL(x), dim=1)[0, 1].item()


# ----------------------------------------------------------------------------------------------
# Faixas de risco
# Cortes: 0,05 = limiar da meta de sensibilidade 0,95 na validação da v7 (modo "sensível");
#         THRESHOLD (0,15 na v7) = limiar da meta 0,90 na validação (modo "equilibrado");
#         0,50 = limiar padrão. Os cortes e as estatísticas abaixo valem para a v7: recalcular ao trocar o checkpoint.
# ----------------------------------------------------------------------------------------------
LIMIAR_SENSIVEL = 0.05
LIMIAR_PADRAO = 0.50
BAND_EDGES = (LIMIAR_SENSIVEL, THRESHOLD, LIMIAR_PADRAO)
TESTE_INTERNO_PREVALENCIA = "14%"  # fração de melanomas no teste interno (HAM10000, dermatoscopia, 1.166 lesões)

# nome, classe css, conduta sugerida, (% das lesões da faixa que eram melanoma, % dos melanomas que caíram na faixa)
BANDS = [
    ("Baixa", "is-b1",
     "O modelo não sinaliza a lesão. Isso não exclui malignidade: manter a avaliação clínica habitual.",
     (0.7, 3.0)),
    ("Média", "is-b2",
     "Zona intermediária. Considerar reavaliação dermatoscópica, comparação com imagens prévias ou revisão por especialista.",
     (13.5, 10.9)),
    ("Alta", "is-b3",
     "Considerar dermatoscopia detalhada e avaliação especializada; a indicação de biópsia segue o julgamento clínico.",
     (19.5, 20.6)),
    ("Muito alta", "is-b4",
     "Alta suspeita pelo modelo. Priorizar avaliação especializada e considerar biópsia, conforme o julgamento clínico.",
     (62.8, 65.5)),
]

RESOLUCAO_MINIMA = 150  # lado mínimo (px) abaixo do qual avisamos; a imagem é reduzida a 224x224 pelo modelo


def faixa_de_risco(score: float) -> int:
    for i, edge in enumerate(BAND_EDGES):
        if score < edge:
            return i
    return len(BAND_EDGES)


def _fmt(x: float, casas: int = 1) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def render_result(score: float, tamanho: tuple[int, int], arquivo: str) -> str:
    """Monta o HTML do cartão de resultado para o profissional de saúde."""
    i = faixa_de_risco(score)
    nome, classe, conduta, (pct_na_faixa, pct_dos_mel) = BANDS[i]
    pct = score * 100

    zonas = []
    inicio = 0.0
    for k, edge in enumerate(list(BAND_EDGES) + [1.0]):
        zonas.append(f'<div class="tn-zone is-b{k + 1}" style="left:{inicio * 100:.2f}%;width:{(edge - inicio) * 100:.2f}%"></div>')
        inicio = edge
    ticks = "".join(
        f'<span class="tn-tick" style="left:{e * 100:.1f}%">{_fmt(e * 100, 0)}%</span>' for e in BAND_EDGES
    )

    avisos = []
    if min(tamanho) < RESOLUCAO_MINIMA:
        avisos.append(
            f"<strong>Baixa resolução</strong> ({tamanho[0]}×{tamanho[1]} px). A estimativa pode ser pouco confiável."
        )
    bloco_avisos = "".join(f'<div class="tn-warn">{a}</div>' for a in avisos)

    if CHECKPOINT_VERSION == "v7":
        estatistica = (
            f'<div class="tn-result-stat">No teste interno do modelo (dermatoscopia, {TESTE_INTERNO_PREVALENCIA} de melanomas), '
            f"<strong>{_fmt(pct_na_faixa)}%</strong> das lesões desta faixa eram melanoma, e a faixa reuniu "
            f"<strong>{_fmt(pct_dos_mel)}%</strong> de todos os melanomas.</div>"
        )
    else:
        estatistica = ""

    registro = (
        f"Registro: modelo {CHECKPOINT_VERSION} · dermatoscopia · pontuação {_fmt(pct)}% · faixa {nome.lower()} · "
        f"cortes {' / '.join(_fmt(e * 100, 0) + '%' for e in BAND_EDGES)} · {arquivo}"
    )

    return f"""
    <div class="tn-card tn-card--elevated tn-result">
      <div class="tn-result-head">
        <div class="tn-result-eyebrow">Estimativa do modelo</div>
        <div class="tn-pill {classe}">Faixa de risco {nome.lower()}</div>
      </div>
      <div>
        <div class="tn-result-number {classe}">{_fmt(pct)}%</div>
        <div class="tn-result-number-label">pontuação do modelo para melanoma (não é uma probabilidade calibrada)</div>
      </div>
      <div class="tn-bar-wrap">
        <div class="tn-bar-track">{''.join(zonas)}<div class="tn-bar-marker" style="left:{pct:.2f}%"></div></div>
        <div class="tn-bar-ticks">{ticks}</div>
      </div>
      {bloco_avisos}
      <div class="tn-divider"></div>
      <div class="tn-result-eyebrow">Conduta a considerar</div>
      <div class="tn-result-orientacao">{conduta}</div>
      {estatistica}
      <div class="tn-result-caption">Apoio à decisão clínica: não substitui o exame clínico nem a histopatologia. Software de pesquisa (TCC), não é dispositivo médico aprovado.</div>
      <div class="tn-result-registro">{registro}</div>
    </div>
    """


def _icon(paths: str, size: int) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f"{paths}</svg>"
    )


_UPLOAD_PATHS = (
    '<path d="M12 15V4"></path><path d="M7.5 8.5 12 4l4.5 4.5"></path>'
    '<path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"></path>'
)
_IMAGE_PATHS = (
    '<rect x="3.5" y="3.5" width="17" height="17" rx="2.5"></rect><circle cx="9" cy="9" r="1.6"></circle>'
    '<path d="m20.5 15-4.5-4.5L5.5 20.5"></path>'
)
_INFO_PATHS = '<circle cx="12" cy="12" r="9"></circle><path d="M12 11v5"></path><path d="M12 7.6v.1"></path>'

EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []


GALLERY_ITEMS = [(str(p), f"Exemplo {i + 1:02d}") for i, p in enumerate(EXAMPLE_FILES)]


def _hero_image_uri() -> str:
    """Foto decorativa do topo, embutida em base64 (não depende de servir arquivo)."""
    if not HERO_IMAGE_PATH.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("ascii")


_HERO_URI = _hero_image_uri()
_HERO_PHOTO = f'<div class="tn-hero-photo" style="background-image:url({_HERO_URI})"></div>' if _HERO_URI else ""

HEADER_HTML = f"""
<nav class="tn-nav">
  <a class="tn-logo" href="#">Tonare</a>
  <div class="tn-nav-links">
    <a href="#como-funciona">Como funciona</a>
    <a class="tn-nav-pill" href="#desempenho">Desempenho</a>
  </div>
</nav>
<section class="tn-hero">
  <div class="tn-hero-art" aria-hidden="true">
    <div class="tn-hero-blob"></div>
    {_HERO_PHOTO}
    <svg class="tn-hero-lines" viewBox="0 0 600 520" fill="none" preserveAspectRatio="none">
      <path d="M150 0 C 80 120, 90 260, 260 300 S 520 360, 600 470"></path>
      <path d="M40 330 C 140 250, 300 260, 380 340"></path>
    </svg>
  </div>
  <div class="tn-kicker">Apoio à decisão clínica</div>
  <h1 class="tn-title">Triagem de <em>melanoma</em><br>por imagem.</h1>
  <p class="tn-subtitle">Envie a dermatoscopia da lesão para obter a estimativa de risco do modelo.</p>
</section>
"""

UPLOAD_HEAD_HTML = f"""
<div class="tn-card-head">
  <span class="tn-icon-badge">{_icon(_UPLOAD_PATHS, 26)}</span>
  <div>
    <h2>Imagem da lesão</h2>
    <p>Dermatoscopia de uma única lesão.</p>
  </div>
</div>
"""

# Desenhado por cima do gr.Image, que continua por baixo recebendo o clique e o arrastar.
DROP_OVERLAY_HTML = f"""
<div class="tn-drop-content">
  <span class="tn-drop-icon">{_icon(_UPLOAD_PATHS, 44)}</span>
  <span class="tn-drop-title">Arraste a imagem aqui</span>
  <span class="tn-drop-or">ou</span>
  <span class="tn-drop-btn">Selecionar imagem</span>
  <span class="tn-drop-help">Formatos aceitos: JPG, JPEG, PNG</span>
</div>
"""

EXAMPLES_HEAD_HTML = f"""
<div class="tn-card-head">
  <span class="tn-icon-badge">{_icon(_IMAGE_PATHS, 24)}</span>
  <div>
    <h2>Casos de exemplo</h2>
    <p>Dermatoscopias sorteadas do conjunto de teste (não entraram no treino). Clique para carregar.</p>
  </div>
</div>
"""

AVISO_HTML = f"""
<div class="tn-aviso">
  {_icon(_INFO_PATHS, 22)}
  <span>Protótipo de pesquisa (TCC). Apoio à decisão do profissional de saúde; não é dispositivo médico e não substitui a avaliação clínica.</span>
</div>
"""

INFO_HTML = """
<section class="tn-info">
  <div id="como-funciona" class="tn-info-item">
    <div class="tn-kicker">Como funciona</div>
    <p>Uma rede neural convolucional (ResNet-50, transfer learning) analisa a imagem e devolve uma <strong>pontuação</strong>
    de 0 a 100% para melanoma contra nevo. A pontuação é agrupada em quatro <strong>faixas de risco</strong>; os cortes vêm da
    validação do modelo (sensibilidade de 95% e de 90%) e do limiar padrão de 50%. O modelo foi treinado com HAM10000,
    MILK10k, BCN20000 e Fitzpatrick17k (tons I–IV).</p>
  </div>
  <div id="desempenho" class="tn-info-item">
    <div class="tn-kicker">Desempenho</div>
    <p>Em dermatoscopia (teste interno, HAM10000) a AUC é 0,92, com sensibilidade 0,86 e especificidade 0,80 no corte de 15%.
    Em testes externos independentes a AUC fica em 0,79–0,93.</p>
  </div>
</section>
"""

PLACEHOLDER_HTML = '<div class="tn-placeholder">Envie uma imagem ou escolha um exemplo para ver a estimativa.</div>'


def on_image_change(path: str | None):
    tem_imagem = bool(path)
    return gr.update(interactive=tem_imagem), gr.update(visible=tem_imagem)


def on_analisar(path: str | None):
    if not path:
        return gr.update(), gr.update(visible=True), gr.update(visible=False)
    img = PILImage.open(path)
    score = predict(img)
    html = render_result(score, img.size, Path(path).name)
    return html, gr.update(visible=False), gr.update(visible=True)


def on_trocar():
    return (
        None,
        gr.update(interactive=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def on_gallery_select(evt: gr.SelectData):
    return str(EXAMPLE_FILES[evt.index])


CSS = CSS_PATH.read_text(encoding="utf-8")
# A paleta só foi desenhada em modo claro: impede o Gradio de trocar para o tema escuro do sistema.
# Sem internet as fontes caem para os fallbacks definidos em tonare.css.
HEAD = (
    # Só mexe na classe quando 'dark' realmente existe e vigia apenas o <body>. Antes o vigia olhava a página inteira e
    # chamava classList.remove('dark') a cada mudança de classe; o próprio remove() reescreve o atributo `class` do body e
    # dispara de novo o vigia, em laço infinito que congela a aba (ao colocar uma imagem, o Gradio muda várias classes).
    "<script>document.addEventListener('DOMContentLoaded', () => {"
    "const tiraDark = () => { if (document.body.classList.contains('dark')) document.body.classList.remove('dark'); };"
    "tiraDark();"
    "new MutationObserver(tiraDark).observe(document.body, {attributes: true, attributeFilter: ['class']});"
    "});</script>"
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;'
    '1,6..72,400&family=Nunito+Sans:opsz,wght@6..12,400;6..12,600&display=swap">'
)

with gr.Blocks(title="Tonare") as demo:
    with gr.Column(elem_classes="tn-page"):
        gr.HTML(HEADER_HTML, elem_classes="tn-header-block")

        with gr.Row(elem_classes="tn-main"):
            with gr.Column(elem_classes="tn-col-primary"):
                with gr.Column(elem_classes=["tn-card", "tn-card--elevated", "tn-upload-card"]):
                    gr.HTML(UPLOAD_HEAD_HTML)
                    with gr.Column(elem_classes="tn-dropwrap"):
                        entrada = gr.Image(
                            type="filepath", show_label=False, sources=["upload"],
                            elem_classes=["tn-dropzone", "tn-photo"],
                        )
                        gr.HTML(DROP_OVERLAY_HTML, elem_classes="tn-drop-overlay")
                    with gr.Row(elem_classes="tn-photo-footer"):
                        botao_trocar = gr.Button("Trocar imagem", elem_classes="tn-btn-secondary", visible=False)
                        botao_analisar = gr.Button("Analisar imagem", elem_classes="tn-btn-primary", interactive=False)

            with gr.Column(elem_classes="tn-col-aside"):
                with gr.Column(visible=True) as exemplos_group:
                    with gr.Column(elem_classes=["tn-card", "tn-card--elevated"]):
                        gr.HTML(EXAMPLES_HEAD_HTML)
                        galeria = gr.Gallery(
                            value=GALLERY_ITEMS, columns=3, show_label=False, allow_preview=False,
                            elem_classes="tn-gallery", object_fit="cover", height="auto",
                        )
                        gr.HTML(AVISO_HTML)
                with gr.Column(visible=False) as resultado_group:
                    resultado_html = gr.HTML(PLACEHOLDER_HTML)

        gr.HTML(INFO_HTML)

    entrada.change(on_image_change, inputs=entrada, outputs=[botao_analisar, botao_trocar])
    botao_analisar.click(on_analisar, inputs=entrada, outputs=[resultado_html, exemplos_group, resultado_group])
    botao_trocar.click(
        on_trocar, inputs=None,
        outputs=[entrada, botao_analisar, botao_trocar, exemplos_group, resultado_group],
    )
    galeria.select(on_gallery_select, inputs=None, outputs=entrada)

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS, head=HEAD, theme=gr.themes.Base())
