import gradio as gr
from .config import *
from .messages import *
from .ui_vote import *
from .ui_battle import *
from .ui_leaderboard import *


with gr.Blocks() as about:
    gr.Markdown(ABOUT)
    gr.DownloadButton("DL", DB_PATH)

with gr.Blocks(css="footer {visibility: hidden}textbox{resize:none}", title="TTS Arena") as app:
    gr.Markdown(DESCR)
    gr.TabbedInterface([vote, battle, leaderboard, about], ['Vote', 'Battle', 'Leaderboard', 'About'])
    if CITATION_TEXT:
        with gr.Row():
            with gr.Accordion("Citation", open=False):
                gr.Markdown(f"If you use this data in your publication, please cite us!\n\nCopy the BibTeX citation to cite this source:\n\n```bibtext\n{CITATION_TEXT}\n```\n\nPlease note that all generated audio clips should be assumed unsuitable for redistribution or commercial use.")
