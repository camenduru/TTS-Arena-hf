import gradio as gr
from .config import *
from .synth import *
from .vote import *
from .messages import *

def disable():
    return [gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)]
def enable():
    return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)]


with gr.Blocks() as vote:
    # sample played
    #aplayed = gr.State(value=False)
    #bplayed = gr.State(value=False)
    # voter ID
    useridstate = gr.State()
    gr.Markdown(INSTR)
    with gr.Group():
        with gr.Row():
            text = gr.Textbox(container=False, show_label=False, placeholder="Enter text to synthesize", lines=1, max_lines=1, scale=9999999, min_width=0)
            randomt = gr.Button('🎲', scale=0, min_width=0, variant='tool')
        randomt.click(randomsent, outputs=[text, randomt])
        btn = gr.Button("Synthesize", variant='primary')
    model1 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=False)
    #model1 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=True)
    model2 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=False)
    #model2 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=True)
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
    nxtroundbtn = gr.Button('Next round', visible=False)
    # outputs = [text, btn, r2, model1, model2, prevmodel1, aud1, prevmodel2, aud2, abetter, bbetter]
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
        nxtroundbtn
    ]
    """
    text,
        "Synthesize",
        gr.update(visible=True), # r2
        mdl1, # model1
        mdl2, # model2
        gr.update(visible=True, value=results[mdl1]), # aud1
        gr.update(visible=True, value=results[mdl2]), # aud2
        gr.update(visible=True, interactive=False), #abetter
        gr.update(visible=True, interactive=False), #bbetter
        gr.update(visible=False), #prevmodel1
        gr.update(visible=False), #prevmodel2
        gr.update(visible=False), #nxt round btn"""
    btn.click(disable, outputs=[btn, abetter, bbetter]).then(synthandreturn, inputs=[text], outputs=outputs).then(enable, outputs=[btn, abetter, bbetter])
    nxtroundbtn.click(clear_stuff, outputs=outputs)

    # Allow interaction with the vote buttons only when both audio samples have finished playing
    #aud1.stop(unlock_vote, outputs=[abetter, bbetter, aplayed, bplayed], inputs=[gr.State(value=0), aplayed, bplayed])
    #aud2.stop(unlock_vote, outputs=[abetter, bbetter, aplayed, bplayed], inputs=[gr.State(value=1), aplayed, bplayed])

    # nxt_outputs = [prevmodel1, prevmodel2, abetter, bbetter]
    nxt_outputs = [abetter, bbetter, prevmodel1, prevmodel2, nxtroundbtn]
    abetter.click(a_is_better, outputs=nxt_outputs, inputs=[model1, model2, useridstate])
    bbetter.click(b_is_better, outputs=nxt_outputs, inputs=[model1, model2, useridstate])
