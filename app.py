DESCR = """
# TTS Arena

Vote on different speech synthesis models!

## Instructions

* Listen to two anonymous models
* Vote on which one is more natural and realistic
* If there's a tie, click Skip

*IMPORTANT: Do not only rank the outputs based on naturalness. Also rank based on intelligibility (can you actually tell what they're saying?) and other factors (does it sound like a human?).*

**When you're ready to begin, click the Start button below!** The model names will be revealed once you vote.
""".strip()
import gradio as gr
import random
import os
from datasets import load_dataset
dataset = load_dataset("ttseval/tts-arena", token=os.getenv('HF_TOKEN'))
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)
model_names = {
    'styletts2': 'StyleTTS 2',
    'tacotron': 'Tacotron',
    'speedyspeech': 'Speedy Speech',
    'overflow': 'Overflow TTS',
    'vits': 'VITS',
    'vitsneon': 'VITS Neon',
    'neuralhmm': 'Neural HMM',
    'glow': 'Glow TTS',
    'fastpitch': 'FastPitch',
}
def get_random_split(existing_split=None):
    choice = random.choice(list(dataset.keys()))
    if existing_split and choice == existing_split:
        return get_random_split(choice)
    else:
        return choice
def get_random_splits():
    choice1 = get_random_split()
    choice2 = get_random_split(choice1)
    return (choice1, choice2)
def a_is_better(model1, model2):
    chosen_model = model1
    print(chosen_model)
    return reload(model1, model2)
def b_is_better(model1, model2):
    chosen_model = model2
    print(chosen_model)
    return reload(model1, model2)
def reload(chosenmodel1=None, chosenmodel2=None):
    # Select random splits
    split1, split2 = get_random_splits()
    d1, d2 = (dataset[split1], dataset[split2])
    choice1, choice2 = (d1.shuffle()[0]['audio'], d2.shuffle()[0]['audio'])
    if split1 in model_names:
        split1 = model_names[split1]
    if split2 in model_names:
        split2 = model_names[split2]
    out = [
        (choice1['sampling_rate'], choice1['array']),
        (choice2['sampling_rate'], choice2['array']),
        split1,
        split2
    ]
    if chosenmodel1: out.append(f'This model was {chosenmodel1}')
    if chosenmodel2: out.append(f'This model was {chosenmodel2}')
    return out
with gr.Blocks(theme=theme, css="footer {visibility: hidden}") as demo:
# with gr.Blocks() as demo:
    gr.Markdown(DESCR)
    with gr.Row():
        gr.HTML('<div align="left"><h3>Model A</h3></div>')
        gr.HTML('<div align="right"><h3>Model B</h3></div>')
    model1 = gr.Textbox(interactive=False, visible=False)
    model2 = gr.Textbox(interactive=False, visible=False)
    with gr.Group():
        with gr.Row():
            prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A")
            prevmodel2 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model B", text_align="right")
        with gr.Row():
            aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
            aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
        with gr.Row():
            abetter = gr.Button("A is Better", scale=3)
            skipbtn = gr.Button("Skip", scale=1)
            bbetter = gr.Button("B is Better", scale=3)
    outputs = [aud1, aud2, model1, model2, prevmodel1, prevmodel2]
    abetter.click(a_is_better, outputs=outputs, inputs=[model1, model2])
    bbetter.click(b_is_better, outputs=outputs, inputs=[model1, model2])
    skipbtn.click(b_is_better, outputs=outputs, inputs=[model1, model2])
    demo.load(reload, outputs=[aud1, aud2, model1, model2])
demo.queue(api_open=False).launch(show_api=False)