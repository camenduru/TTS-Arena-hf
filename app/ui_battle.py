import gradio as gr
from .config import *
from .ui import *
from .synth import *
from .vote import *
from .messages import *

def disable():
    return [gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)]
def enable():
    return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)]


with gr.Blocks() as battle:
    battle_useridstate = gr.State()
    
    gr.Markdown(BATTLE_INSTR)
    model1 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=False)
    model2 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=False)
    with gr.Group():
        with gr.Row():
            text = gr.Textbox(container=False, show_label=False, placeholder="Enter text to synthesize", lines=1, max_lines=1, scale=9999999, min_width=0)
            randomt_battle = gr.Button('🎲', scale=0, min_width=0, variant='tool')
    with gr.Row():
        with gr.Column(scale=10):
            model1s = gr.Dropdown(label="Model 1", container=False, show_label=False, choices=AVAILABLE_MODELS.keys(), interactive=True, value=list(AVAILABLE_MODELS.keys())[0])
        with gr.Column(scale=10):
            model2s = gr.Dropdown(label="Model 2", container=False, show_label=False, choices=AVAILABLE_MODELS.keys(), interactive=True, value=list(AVAILABLE_MODELS.keys())[1])
    randomt_battle.click(randomsent_battle, outputs=[text, randomt_battle, model1s, model2s])
    btn = gr.Button("Synthesize", variant='primary')
    with gr.Row(visible=False) as r2:
        with gr.Column():
            with gr.Group():
                aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False)
                abetter = gr.Button("A is better", variant='primary')
                prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A", text_align="center", lines=1, max_lines=1, visible=False)
        with gr.Column():
            with gr.Group():
                aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False)
                bbetter = gr.Button("B is better", variant='primary')
                prevmodel2 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model B", text_align="center", lines=1, max_lines=1, visible=False)
    outputs = [
        text,
        btn,
        r2,
        model1,
        model2,
        aud1,
        aud2,
        abetter,
        bbetter,
        prevmodel1,
        prevmodel2,
    ]
    btn.click(disable, outputs=[btn, abetter, bbetter]).then(synthandreturn_battle, inputs=[text, model1s, model2s], outputs=outputs).then(enable, outputs=[btn, abetter, bbetter])
    nxt_outputs = [abetter, bbetter, prevmodel1, prevmodel2]
    abetter.click(a_is_better_battle, outputs=nxt_outputs, inputs=[model1, model2, battle_useridstate])
    bbetter.click(b_is_better_battle, outputs=nxt_outputs, inputs=[model1, model2, battle_useridstate])
    battle.load(random_m, outputs=[model1s, model2s])
