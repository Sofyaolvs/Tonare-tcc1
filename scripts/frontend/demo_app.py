"""
Interface de demonstração do Tonare — pra mostrar ao vivo no pitch.

Implementa a especificação de design fechada com a IA de design (paleta, tipografia,
layout de 2 colunas, cartão de resultado com barra de probabilidade e marcador de limiar).
Ver scripts/frontend/tonare.css para os tokens e classes.

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


MODEL, THRESHOLD = load_model()  # THRESHOLD é a constante configurável pedida na especificação
print(f"Modelo carregado. Limiar de decisão (escolhido na validação): {THRESHOLD:.2f}")


def predict(image: PILImage.Image) -> float:
    """predict(img) -> probabilidade de melanoma (0 a 1)."""
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)
        return F.softmax(MODEL(x), dim=1)[0, 1].item()


def render_result(prob: float, limiar: float) -> str:
    """Monta o HTML do cartão de resultado a partir da probabilidade e do limiar."""
    suspeito = prob >= limiar
    estado = "is-suspeito" if suspeito else "is-benigno"
    pill_texto = "Suspeito de melanoma" if suspeito else "Padrão de nevo benigno"
    cor = "var(--tn-sus)" if suspeito else "var(--tn-ben)"

    if suspeito:
        orientacao = "Recomenda-se encaminhamento para avaliação dermatológica."
        limiar_fmt = f"{limiar:.2f}".replace(".", ",")
        legenda = (
            f"Limiar de decisão: {limiar_fmt}. Probabilidades iguais ou acima desse valor são "
            "classificadas como suspeitas. Resultado de triagem — não substitui avaliação médica."
        )
    else:
        orientacao = "Sem sinais de suspeita neste limiar. Mantenha o acompanhamento de rotina da pele."
        limiar_fmt = f"{limiar:.2f}".replace(".", ",")
        legenda = (
            f"Limiar de decisão: {limiar_fmt}. Probabilidades abaixo desse valor são "
            "classificadas como padrão benigno. Em caso de mudança de cor, forma ou tamanho, "
            "procure um dermatologista."
        )

    pct = prob * 100
    pct_fmt = f"{pct:.1f}".replace(".", ",")
    limiar_pct = limiar * 100

    return f"""
    <div class="tn-card tn-card--elevated tn-result">
      <div class="tn-result-head">
        <div class="tn-result-eyebrow">Resultado da análise</div>
        <div class="tn-pill {estado}">{pill_texto}</div>
      </div>
      <div>
        <div class="tn-result-number" style="color:{cor}">{pct_fmt}%</div>
        <div class="tn-result-number-label">probabilidade estimada de melanoma</div>
      </div>
      <div>
        <div class="tn-bar-track {estado}">
          <div class="tn-bar-fill {estado}" style="width:{pct:.2f}%"></div>
          <div class="tn-bar-marker" style="left:{limiar_pct:.2f}%"></div>
        </div>
        <div class="tn-bar-labels">
          <span>0%</span><span>limiar {limiar_fmt}</span><span>100%</span>
        </div>
      </div>
      <div class="tn-divider"></div>
      <div class="tn-result-orientacao">{orientacao}</div>
      <div class="tn-result-caption">{legenda}</div>
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
    <a class="tn-nav-pill" href="#sobre">Sobre o projeto</a>
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
  <div class="tn-kicker">Triagem de pele por imagem</div>
  <h1 class="tn-title">Um novo olhar para<br>o cuidado com a <em>pele.</em></h1>
  <p class="tn-subtitle">Envie uma imagem para iniciar a triagem.</p>
</section>
"""

UPLOAD_HEAD_HTML = f"""
<div class="tn-card-head">
  <span class="tn-icon-badge">{_icon(_UPLOAD_PATHS, 26)}</span>
  <h2>Envie sua fotografia</h2>
</div>
"""

# Desenhado por cima do gr.Image, que continua por baixo recebendo o clique e o arrastar.
DROP_OVERLAY_HTML = f"""
<div class="tn-drop-content">
  <span class="tn-drop-icon">{_icon(_UPLOAD_PATHS, 44)}</span>
  <span class="tn-drop-title">Arraste sua imagem aqui</span>
  <span class="tn-drop-or">ou</span>
  <span class="tn-drop-btn">Selecionar imagem</span>
  <span class="tn-drop-help">Formatos aceitos: JPG, JPEG, PNG &nbsp;•&nbsp; Tamanho máximo: 10 MB</span>
</div>
"""

EXAMPLES_HEAD_HTML = f"""
<div class="tn-card-head">
  <span class="tn-icon-badge">{_icon(_IMAGE_PATHS, 24)}</span>
  <div>
    <h2>Experimente um exemplo</h2>
    <p>Clique em uma imagem para testar a ferramenta.</p>
  </div>
</div>
"""

AVISO_HTML = f"""
<div class="tn-aviso">
  {_icon(_INFO_PATHS, 22)}
  <span>Protótipo de pesquisa (TCC). Apoio à triagem;<br>não substitui avaliação médica.</span>
</div>
"""

INFO_HTML = """
<section class="tn-info">
  <div id="como-funciona" class="tn-info-item">
    <div class="tn-kicker">Como funciona</div>
    <p>Uma rede neural convolucional (ResNet-50 com transfer learning) analisa a imagem da lesão e estima a
    probabilidade de melanoma. Quando ela fica no limiar de decisão ou acima dele (limiar escolhido na
    validação), a lesão é sinalizada como suspeita.</p>
  </div>
  <div id="sobre" class="tn-info-item">
    <div class="tn-kicker">Sobre o projeto</div>
    <p>Trabalho de Conclusão de Curso sobre triagem de melanoma por imagem, com análise de equidade do modelo
    entre tons de pele. É uma ferramenta de apoio à triagem, não um dispositivo médico nem um diagnóstico.</p>
  </div>
</section>
"""

PLACEHOLDER_HTML = '<div class="tn-placeholder">Envie uma foto ou escolha um exemplo para ver a análise.</div>'


def on_image_change(path: str | None):
    tem_imagem = bool(path)
    return gr.update(interactive=tem_imagem), gr.update(visible=tem_imagem)


def on_analisar(path: str | None):
    if not path:
        return gr.update(), gr.update(visible=True), gr.update(visible=False)
    img = PILImage.open(path)
    prob = predict(img)
    html = render_result(prob, THRESHOLD)
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
    "<script>new MutationObserver(() => document.body && document.body.classList.remove('dark'))"
    ".observe(document.documentElement, {subtree: true, attributes: true, attributeFilter: ['class']});</script>"
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
