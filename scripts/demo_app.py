"""
Interface de demonstração do Tonare — pra mostrar ao vivo no pitch.

Implementa a especificação de design fechada com a IA de design (paleta, tipografia,
layout de 2 colunas, cartão de resultado com barra de probabilidade e marcador de limiar).
Ver scripts/tonare.css para os tokens e classes.

Uso:
    python scripts/demo_app.py
Depois abre http://127.0.0.1:7860 no navegador (o Gradio também tenta abrir sozinho).
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "data" / "processed" / "model_resnet50_baseline_v2.pt"
EXAMPLES_DIR = BASE_DIR / "data" / "demo_examples"
CSS_PATH = Path(__file__).resolve().parent / "tonare.css"

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
    limiar_pct = limiar * 100

    return f"""
    <div class="tn-card tn-card--elevated tn-result">
      <div class="tn-result-head">
        <div class="tn-result-eyebrow">Resultado da análise</div>
        <div class="tn-pill {estado}">{pill_texto}</div>
      </div>
      <div>
        <div class="tn-result-number" style="color:{cor}">{pct:.1f}%</div>
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


UPLOAD_ICON = (
    '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 15V4"></path><path d="M7.5 8.5 12 4l4.5 4.5"></path>'
    '<path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"></path></svg>'
)

HEADER_HTML = f"""
<div class="tn-header">
  <div class="tn-header-left">
    <div class="tn-kicker">Triagem de melanoma por imagem</div>
    <h1 class="tn-title">Tonare</h1>
    <p class="tn-subtitle">Envie uma fotografia de uma lesão de pele para estimar a probabilidade de melanoma.</p>
  </div>
  <div class="tn-aviso">Demonstração de pesquisa (TCC) — não é um dispositivo médico. Ferramenta de triagem e apoio à decisão, nunca de diagnóstico.</div>
</div>
"""

PLACEHOLDER_HTML = '<div class="tn-placeholder">Envie uma foto ou escolha um exemplo para ver a análise.</div>'

EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
GALLERY_ITEMS = [(str(p), f"Exemplo {i + 1:02d}") for i, p in enumerate(EXAMPLE_FILES)]


def on_image_change(path: str | None):
    tem_imagem = bool(path)
    nome = f"<span>{Path(path).name}</span>" if tem_imagem else ""
    return (
        nome,
        gr.update(interactive=tem_imagem),
        gr.update(interactive=tem_imagem),
    )


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
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def on_gallery_select(evt: gr.SelectData):
    return str(EXAMPLE_FILES[evt.index])


CSS = CSS_PATH.read_text(encoding="utf-8")

with gr.Blocks(title="Tonare") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(elem_classes="tn-main"):
        with gr.Column(elem_classes="tn-col-primary"):
            with gr.Column(elem_classes=["tn-card", "tn-card--elevated"]):
                gr.HTML('<p class="tn-card-title">Fotografia da lesão</p>')
                entrada = gr.Image(type="filepath", show_label=False, elem_classes=["tn-dropzone", "tn-photo"])
                with gr.Row(elem_classes="tn-photo-footer"):
                    nome_arquivo = gr.HTML("")
                    with gr.Row(elem_classes="tn-btn-row"):
                        botao_trocar = gr.Button("Trocar imagem", elem_classes="tn-btn-secondary", interactive=False)
                        botao_analisar = gr.Button("Analisar", elem_classes="tn-btn-primary", interactive=False)

        with gr.Column(elem_classes="tn-col-aside"):
            with gr.Column(visible=True) as exemplos_group:
                with gr.Column(elem_classes="tn-card"):
                    gr.HTML(
                        '<div class="tn-card-heading"><h2>Exemplos</h2>'
                        '<p>Clique em uma imagem para testar a ferramenta.</p></div>'
                    )
                    galeria = gr.Gallery(
                        value=GALLERY_ITEMS, columns=3, show_label=False, allow_preview=False,
                        elem_classes="tn-gallery", object_fit="cover",
                    )
            with gr.Column(visible=False) as resultado_group:
                resultado_html = gr.HTML(PLACEHOLDER_HTML)

    entrada.change(on_image_change, inputs=entrada, outputs=[nome_arquivo, botao_analisar, botao_trocar])
    botao_analisar.click(on_analisar, inputs=entrada, outputs=[resultado_html, exemplos_group, resultado_group])
    botao_trocar.click(
        on_trocar, inputs=None,
        outputs=[entrada, nome_arquivo, botao_analisar, botao_trocar, exemplos_group, resultado_group],
    )
    galeria.select(on_gallery_select, inputs=None, outputs=entrada)

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS, theme=gr.themes.Base())
