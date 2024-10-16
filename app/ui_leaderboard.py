import gradio as gr
from .config import *
from .leaderboard import *
from .messages import *

with gr.Blocks() as leaderboard:
    gr.Markdown(LDESC)
    df = gr.Dataframe(interactive=False, min_width=0, wrap=True, column_widths=[30, 200, 50, 50])
    reloadbtn = gr.Button("Refresh")
    with gr.Row():
        reveal_prelim = gr.Checkbox(label="Reveal preliminary results", info="Show all models, including models with very few human ratings.", scale=1)
        hide_battle_votes = gr.Checkbox(label="Hide Battle Mode votes", info="Exclude votes obtained through Battle Mode.", scale=1)
    reveal_prelim.input(get_leaderboard, inputs=[reveal_prelim, hide_battle_votes], outputs=[df])
    hide_battle_votes.input(get_leaderboard, inputs=[reveal_prelim, hide_battle_votes], outputs=[df])
    leaderboard.load(get_leaderboard, inputs=[reveal_prelim, hide_battle_votes], outputs=[df])
    reloadbtn.click(get_leaderboard, inputs=[reveal_prelim, hide_battle_votes], outputs=[df])
    # gr.Markdown("DISCLAIMER: The licenses listed may not be accurate or up to date, you are responsible for checking the licenses before using the models. Also note that some models may have additional usage restrictions.")
