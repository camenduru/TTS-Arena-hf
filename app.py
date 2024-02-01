import gradio as gr
import random
import os
import shutil
import pandas as pd
import sqlite3
from datasets import load_dataset
import threading
import time
from huggingface_hub import HfApi


DESCR = """
# TTS Arena

Vote on different speech synthesis models!
""".strip()
INSTR = """
## Instructions

* Listen to two anonymous models
* Vote on which one is more natural and realistic, with better prosody and intonation
* If there's a tie, click Skip

**When you're ready to begin, click the Start button below!** The model names will be revealed once you vote.
""".strip()
request = ''
if os.getenv('HF_ID'):
    request = f"""
### Request Model

Please fill out [this form](https://huggingface.co/spaces/{os.getenv('HF_ID')}/discussions/new?title=%5BModel+Request%5D+&description=%23%23%20Model%20Request%0A%0A%2A%2AModel%20website%2Fpaper%20%28if%20applicable%29%2A%2A%3A%0A%2A%2AModel%20available%20on%2A%2A%3A%20%28coqui%7CHF%20pipeline%7Ccustom%20code%29%0A%2A%2AWhy%20do%20you%20want%20this%20model%20added%3F%2A%2A%0A%2A%2AComments%3A%2A%2A) to request a model.
"""
ABOUT = f"""
## About

The TTS Arena is a project created to evaluate leading speech synthesis models. It is inspired by the [Chatbot Arena](https://chat.lmsys.org/) by LMSYS.

### How it Works

First, vote on two samples of text-to-speech models. The models that synthesized the samples are not revealed to mitigate bias.

As you vote, the leaderboard will be updated based on votes. We calculate a score for each model using a method similar to the [Elo system](https://en.wikipedia.org/wiki/Elo_rating_system).

### Motivation

Recently, many new open-access speech synthesis models have been made available to the community. However, there is no standardized evaluation or benchmark to measure the quality and naturalness of these models.

The TTS Arena is an attempt to benchmark these models and find the highest-quality models available to the community.

{request}
""".strip()
LDESC = """
## Leaderboard

A list of the models, based on how highly they are ranked!
""".strip()


dataset = load_dataset("ttseval/tts-arena-new", token=os.getenv('HF_TOKEN'))
def reload_db():
    global dataset
    dataset = load_dataset("ttseval/tts-arena-new", token=os.getenv('HF_TOKEN'))
    return 'Reload Dataset'
def del_db(txt):
    if not txt.lower() == 'delete db':
        raise gr.Error('You did not enter "delete db"')
    api = HfApi(
        token=os.getenv('HF_TOKEN')
    )
    os.remove('database.db')
    create_db()
    api.delete_file(
        path_in_repo='database.db',
        repo_id=os.getenv('DATASET_ID'),
        repo_type='dataset'
    )
    return 'Delete DB'
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)
model_names = {
    'styletts2': 'StyleTTS 2',
    'tacotron': 'Tacotron',
    'tacotronph': 'Tacotron Phoneme',
    'tacotrondca': 'Tacotron DCA',
    'speedyspeech': 'Speedy Speech',
    'overflow': 'Overflow TTS',
    'vits': 'VITS',
    'vitsneon': 'VITS Neon',
    'neuralhmm': 'Neural HMM',
    'glow': 'Glow TTS',
    'fastpitch': 'FastPitch',
    'jenny': 'Jenny',
    'tortoise': 'Tortoise TTS',
    'xtts2': 'Coqui XTTSv2',
    'xtts': 'Coqui XTTS',
    'openvoice': 'MyShell OpenVoice',
    'elevenlabs': 'ElevenLabs',
    'openai': 'OpenAI',
    'hierspeech': 'HierSpeech++',
    'pheme': 'PolyAI Pheme',
    'speecht5': 'SpeechT5',
}
model_licenses = {
    'styletts2': 'MIT',
    'tacotron': 'BSD-3',
    'tacotronph': 'BSD-3',
    'tacotrondca': 'BSD-3',
    'speedyspeech': 'BSD-3',
    'overflow': 'MIT',
    'vits': 'MIT',
    'openvoice': 'MIT',
    'vitsneon': 'BSD-3',
    'neuralhmm': 'MIT',
    'glow': 'MIT',
    'fastpitch': 'Apache 2.0',
    'jenny': 'Jenny License',
    'tortoise': 'Apache 2.0',
    'xtts2': 'CPML (NC)',
    'xtts': 'CPML (NC)',
    'elevenlabs': 'Proprietary',
    'openai': 'Proprietary',
    'hierspeech': 'MIT',
    'pheme': 'CC-BY',
    'speecht5': 'MIT',
}
# def get_random_split(existing_split=None):
#     choice = random.choice(list(dataset.keys()))
#     if existing_split and choice == existing_split:
#         return get_random_split(choice)
#     else:
#         return choice
def get_db():
    return sqlite3.connect('database.db')
def create_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model (
            name TEXT UNIQUE,
            upvote INTEGER,
            downvote INTEGER
        );
    ''')

def get_data():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT name, upvote, downvote FROM model WHERE (upvote + downvote) > 5')
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['name', 'upvote', 'downvote'])
    df['license'] = df['name'].replace(model_licenses)
    df['name'] = df['name'].replace(model_names)
    df['votes'] = df['upvote'] + df['downvote']
    # df['score'] = round((df['upvote'] / df['votes']) * 100, 2) # Percentage score

    ## ELO SCORE
    df['score'] = 1200
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                expected_a = 1 / (1 + 10 ** ((df['score'][j] - df['score'][i]) / 400))
                expected_b = 1 / (1 + 10 ** ((df['score'][i] - df['score'][j]) / 400))
                actual_a = df['upvote'][i] / df['votes'][i]
                actual_b = df['upvote'][j] / df['votes'][j]
                df.at[i, 'score'] += 32 * (actual_a - expected_a)
                df.at[j, 'score'] += 32 * (actual_b - expected_b)
    df['score'] = round(df['score'])
    ## ELO SCORE
    df = df.sort_values(by='score', ascending=False)
    df['order'] = ['#' + str(i + 1) for i in range(len(df))]
    # df = df[['name', 'score', 'upvote', 'votes']]
    df = df[['order', 'name', 'score', 'license', 'votes']]
    return df

# def get_random_splits():
#     choice1 = get_random_split()
#     choice2 = get_random_split(choice1)
#     return (choice1, choice2)
def upvote_model(model):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE model SET upvote = upvote + 1 WHERE name = ?', (model,))
    if cursor.rowcount == 0:
        cursor.execute('INSERT OR REPLACE INTO model (name, upvote, downvote) VALUES (?, 1, 0)', (model,))
    conn.commit()
    cursor.close()
def downvote_model(model):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE model SET downvote = downvote + 1 WHERE name = ?', (model,))
    if cursor.rowcount == 0:
        cursor.execute('INSERT OR REPLACE INTO model (name, upvote, downvote) VALUES (?, 0, 1)', (model,))
    conn.commit()
    cursor.close()
def a_is_better(model1, model2):
    if model1 and model2:
        upvote_model(model1)
        downvote_model(model2)
    return reload(model1, model2)
def b_is_better(model1, model2):
    if model1 and model2:
        upvote_model(model2)
        downvote_model(model1)
    return reload(model1, model2)
def both_bad(model1, model2):
    if model1 and model2:
        downvote_model(model1)
        downvote_model(model2)
    return reload(model1, model2)
def both_good(model1, model2):
    if model1 and model2:
        upvote_model(model1)
        upvote_model(model2)
    return reload(model1, model2)
def reload(chosenmodel1=None, chosenmodel2=None):
    # Select random splits
    row = random.choice(list(dataset['train']))
    options = list(random.choice(list(dataset['train'])).keys())
    split1, split2 = random.sample(options, 2)
    choice1, choice2 = (row[split1], row[split2])
    if chosenmodel1 in model_names:
        chosenmodel1 = model_names[chosenmodel1]
    if chosenmodel2 in model_names:
        chosenmodel2 = model_names[chosenmodel2]
    out = [
        (choice1['sampling_rate'], choice1['array']),
        (choice2['sampling_rate'], choice2['array']),
        split1,
        split2
    ]
    if chosenmodel1: out.append(f'This model was {chosenmodel1}')
    if chosenmodel2: out.append(f'This model was {chosenmodel2}')
    return out

with gr.Blocks() as leaderboard:
    gr.Markdown(LDESC)
    # df = gr.Dataframe(interactive=False, value=get_data())
    df = gr.Dataframe(interactive=False, min_width=0, wrap=True, column_widths=[30, 200, 50, 75, 50])
    reloadbtn = gr.Button("Refresh")
    leaderboard.load(get_data, outputs=[df])
    reloadbtn.click(get_data, outputs=[df])
    gr.Markdown("DISCLAIMER: The licenses listed may not be accurate or up to date, you are responsible for checking the licenses before using the models. Also note that some models may have additional usage restrictions.")

with gr.Blocks() as vote:
    gr.Markdown(INSTR)
    with gr.Row():
        gr.HTML('<div align="left"><h3>Model A</h3></div>')
        gr.HTML('<div align="right"><h3>Model B</h3></div>')
    model1 = gr.Textbox(interactive=False, visible=False)
    model2 = gr.Textbox(interactive=False, visible=False)
    # with gr.Group():
    #     with gr.Row():
    #         prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A")
    #         prevmodel2 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model B", text_align="right")
    #     with gr.Row():
    #         aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
    #         aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
    with gr.Group():
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A")
                    aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
            with gr.Column():
                with gr.Group():
                    prevmodel2 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model B", text_align="right")
                    aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})


    with gr.Row():
        abetter = gr.Button("A is Better", variant='primary', scale=4)
        # skipbtn = gr.Button("Skip", scale=1)
        bbetter = gr.Button("B is Better", variant='primary', scale=4)
    with gr.Row():
        bothbad = gr.Button("Both are Bad", scale=2)
        skipbtn = gr.Button("Skip", scale=1)
        bothgood = gr.Button("Both are Good", scale=2)
    outputs = [aud1, aud2, model1, model2, prevmodel1, prevmodel2]
    abetter.click(a_is_better, outputs=outputs, inputs=[model1, model2])
    bbetter.click(b_is_better, outputs=outputs, inputs=[model1, model2])
    skipbtn.click(b_is_better, outputs=outputs, inputs=[model1, model2])

    bothbad.click(both_bad, outputs=outputs, inputs=[model1, model2])
    bothgood.click(both_good, outputs=outputs, inputs=[model1, model2])

    vote.load(reload, outputs=[aud1, aud2, model1, model2])
with gr.Blocks() as about:
    gr.Markdown(ABOUT)
with gr.Blocks() as admin:
    rdb = gr.Button("Reload Dataset")
    rdb.click(reload_db, outputs=rdb)
    with gr.Group():
        dbtext = gr.Textbox(label="Type \"delete db\" to confirm", placeholder="delete db")
        ddb = gr.Button("Delete DB")
    ddb.click(del_db, inputs=dbtext, outputs=ddb)
with gr.Blocks(theme=theme, css="footer {visibility: hidden}", title="TTS Leaderboard") as demo:
    gr.Markdown(DESCR)
    gr.TabbedInterface([vote, leaderboard, about, admin], ['Vote', 'Leaderboard', 'About', 'Admin (ONLY IN BETA)'])
def restart_space():
    api = HfApi(
        token=os.getenv('HF_TOKEN')
    )
    time.sleep(60 * 60) # Every hour
    print("Syncing DB before restarting space")
    api.upload_file(
        path_or_fileobj='database.db',
        path_in_repo='database.db',
        repo_id=os.getenv('DATASET_ID'),
        repo_type='dataset'
    )
    print("Restarting space")
    api.restart_space(repo_id=os.getenv('HF_ID'))
def sync_db():
    api = HfApi(
        token=os.getenv('HF_TOKEN')
    )
    while True:
        time.sleep(60 * 10)
        print("Uploading DB")
        api.upload_file(
            path_or_fileobj='database.db',
            path_in_repo='database.db',
            repo_id=os.getenv('DATASET_ID'),
            repo_type='dataset'
        )
if os.getenv('HF_ID'):
    restart_thread = threading.Thread(target=restart_space)
    restart_thread.daemon = True
    restart_thread.start()
if os.getenv('DATASET_ID'):
    # Fetch DB
    api = HfApi(
        token=os.getenv('HF_TOKEN')
    )
    print("Downloading DB...")
    try:
        path = api.hf_hub_download(
            repo_id=os.getenv('DATASET_ID'),
            repo_type='dataset',
            filename='database.db',
            cache_dir='./'
        )
        shutil.copyfile(path, 'database.db')
        print("Downloaded DB")
    except:
        pass
    # Update DB
    db_thread = threading.Thread(target=sync_db)
    db_thread.daemon = True
    db_thread.start()
create_db()
demo.queue(api_open=False).launch(show_api=False)