import gradio as gr
import random
import os
import shutil
import pandas as pd
import sqlite3
from datasets import load_dataset
import threading
import time
import uuid
from pathlib import Path
from huggingface_hub import CommitScheduler, delete_file, hf_hub_download
from gradio_client import Client

####################################
# Constants
####################################

AVAILABLE_MODELS = {
    'XTTS': 'xttsv2',
    'WhisperSpeech': 'whisperspeech',
    'ElevenLabs': 'eleven',
    'OpenVoice': 'openvoice',
    'Pheme': 'pheme',
}

SPACE_ID = os.getenv('HF_ID')
MAX_SAMPLE_TXT_LENGTH = 150
DB_DATASET_ID = os.getenv('DATASET_ID')
DB_NAME = "database.db"

# If /data available => means local storage is enabled => let's use it!
DB_PATH = f"/data/{DB_NAME}" if os.path.isdir("/data") else DB_NAME
print(f"Using {DB_PATH}")
# AUDIO_DATASET_ID = "ttseval/tts-arena-new"
CITATION_TEXT = """@misc{tts-arena,
	title        = {Text to Speech Arena},
	author       = {mrfakename and Srivastav, Vaibhav and Pouget, Lucain and Fourrier, Clémentine},
	year         = 2024,
	publisher    = {Hugging Face},
	howpublished = "\\url{https://huggingface.co/spaces/ttseval/TTS-Arena}"
}"""

####################################
# Functions
####################################

def create_db_if_missing():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model (
            name TEXT UNIQUE,
            upvote INTEGER,
            downvote INTEGER
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vote (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            model TEXT,
            vote INTEGER
        );
    ''')
def get_db():
    return sqlite3.connect(DB_PATH)

def get_leaderboard():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT name, upvote, downvote FROM model WHERE (upvote + downvote) > 5')
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['name', 'upvote', 'downvote'])
    df['license'] = df['name'].map(model_licenses).fillna("Unknown")
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


####################################
# Space initialization
####################################

# Download existing DB
if not os.path.isfile(DB_PATH):
    print("Downloading DB...")
    try:
        cache_path = hf_hub_download(repo_id=DB_DATASET_ID, repo_type='dataset', filename=DB_NAME)
        shutil.copyfile(cache_path, DB_PATH)
        print("Downloaded DB")
    except Exception as e:
        print("Error while downloading DB:", e)

# Create DB table (if doesn't exist)
create_db_if_missing()
    
# Sync local DB with remote repo every 5 minute (only if a change is detected)
scheduler = CommitScheduler(
    repo_id=DB_DATASET_ID,
    repo_type="dataset",
    folder_path=Path(DB_PATH).parent,
    every=5,
    allow_patterns=DB_NAME,
)

# Load audio dataset
# audio_dataset = load_dataset(AUDIO_DATASET_ID)

####################################
# Router API
####################################
router = Client("ttseval/tts-router", hf_token=os.getenv('HF_TOKEN'))
####################################
# Gradio app
####################################
MUST_BE_LOGGEDIN = "Please login with Hugging Face to participate in the TTS Arena."
DESCR = """
# TTS Arena

Vote on different speech synthesis models!
""".strip()
# INSTR = """
# ## Instructions

# * Listen to two anonymous models
# * Vote on which synthesized audio sounds more natural to you
# * If there's a tie, click Skip

# **When you're ready to begin, login and begin voting!** The model names will be revealed once you vote.
# """.strip()
INSTR = """
## Instructions

* Enter text to synthesize
* Listen to the two audio clips
* Vote on which synthesized audio sounds more natural to you
* Repeat!

**When you're ready to begin, enter text!** The model names will be revealed once you vote.
""".strip()
request = ''
if SPACE_ID:
    request = f"""
### Request Model

Please fill out [this form](https://huggingface.co/spaces/{SPACE_ID}/discussions/new?title=%5BModel+Request%5D+&description=%23%23%20Model%20Request%0A%0A%2A%2AModel%20website%2Fpaper%20%28if%20applicable%29%2A%2A%3A%0A%2A%2AModel%20available%20on%2A%2A%3A%20%28coqui%7CHF%20pipeline%7Ccustom%20code%29%0A%2A%2AWhy%20do%20you%20want%20this%20model%20added%3F%2A%2A%0A%2A%2AComments%3A%2A%2A) to request a model.
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

### Privacy Statement

We may store text you enter and generated audio. We store a unique ID for each session.

### License

Please assume all generated audio clips are not licensed to be redistributed and may only be used for personal, non-commercial use.
""".strip()
LDESC = """
## Leaderboard

A list of the models, based on how highly they are ranked!
""".strip()




# def reload_audio_dataset():
#     global audio_dataset
#     audio_dataset = load_dataset(AUDIO_DATASET_ID)
#     return 'Reload Audio Dataset'

def del_db(txt):
    if not txt.lower() == 'delete db':
        raise gr.Error('You did not enter "delete db"')

    # Delete local + remote
    os.remove(DB_PATH)
    delete_file(path_in_repo=DB_NAME, repo_id=DB_DATASET_ID, repo_type='dataset')

    # Recreate
    create_db_if_missing()
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
    'metavoice': 'MetaVoice-1B',
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
    'metavoice': 'Apache 2.0',
    'elevenlabs': 'Proprietary',
}
model_links = {
    'styletts2': 'https://github.com/yl4579/StyleTTS2',
    'tacotron': 'https://github.com/NVIDIA/tacotron2',
    'speedyspeech': 'https://github.com/janvainer/speedyspeech',
    'overflow': 'https://github.com/shivammehta25/OverFlow',
    'vits': 'https://github.com/jaywalnut310/vits',
    'openvoice': 'https://github.com/myshell-ai/OpenVoice',
    'neuralhmm': 'https://github.com/ketranm/neuralHMM',
    'glow': 'https://github.com/jaywalnut310/glow-tts',
    'fastpitch': 'https://fastpitch.github.io/',
    'tortoise': 'https://github.com/neonbjb/tortoise-tts',
    'xtts2': 'https://huggingface.co/coqui/XTTS-v2',
    'xtts': 'https://huggingface.co/coqui/XTTS-v1',
    'elevenlabs': 'https://elevenlabs.io/',
    'openai': 'https://help.openai.com/en/articles/8555505-tts-api',
    'hierspeech': 'https://github.com/sh-lee-prml/HierSpeechpp',
    'pheme': 'https://github.com/PolyAI-LDN/pheme',
    'speecht5': 'https://github.com/microsoft/SpeechT5',
    'metavoice': 'https://github.com/metavoiceio/metavoice-src',
}
# def get_random_split(existing_split=None):
#     choice = random.choice(list(audio_dataset.keys()))
#     if existing_split and choice == existing_split:
#         return get_random_split(choice)
#     else:
#         return choice

# def get_random_splits():
#     choice1 = get_random_split()
#     choice2 = get_random_split(choice1)
#     return (choice1, choice2)
def mkuuid(uid):
    if not uid:
        uid = uuid.uuid4()
    return uid
def upvote_model(model, uname):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE model SET upvote = upvote + 1 WHERE name = ?', (model,))
    if cursor.rowcount == 0:
        cursor.execute('INSERT OR REPLACE INTO model (name, upvote, downvote) VALUES (?, 1, 0)', (model,))
    cursor.execute('INSERT INTO vote (username, model, vote) VALUES (?, ?, ?)', (uname, model, 1,))
    with scheduler.lock:
        conn.commit()
    cursor.close()

def downvote_model(model, uname):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE model SET downvote = downvote + 1 WHERE name = ?', (model,))
    if cursor.rowcount == 0:
        cursor.execute('INSERT OR REPLACE INTO model (name, upvote, downvote) VALUES (?, 0, 1)', (model,))
    cursor.execute('INSERT INTO vote (username, model, vote) VALUES (?, ?, ?)', (uname, model, -1,))
    with scheduler.lock:
        conn.commit()
    cursor.close()

def a_is_better(model1, model2, userid):
    userid = mkuuid(userid)
    if model1 and model2:
        upvote_model(model1, str(userid))
        downvote_model(model2, str(userid))
    return reload(model1, model2, userid, chose_a=True)
def b_is_better(model1, model2, userid):
    userid = mkuuid(userid)
    if model1 and model2:
        upvote_model(model2, str(userid))
        downvote_model(model1, str(userid))
    return reload(model1, model2, userid, chose_b=True)
def both_bad(model1, model2, userid):
    userid = mkuuid(userid)
    if model1 and model2:
        downvote_model(model1, str(userid))
        downvote_model(model2, str(userid))
    return reload(model1, model2, userid)
def both_good(model1, model2, userid):
    userid = mkuuid(userid)
    if model1 and model2:
        upvote_model(model1, str(userid))
        upvote_model(model2, str(userid))
    return reload(model1, model2, userid)
def reload(chosenmodel1=None, chosenmodel2=None, userid=None, chose_a=False, chose_b=False):
    # Select random splits
    # row = random.choice(list(audio_dataset['train']))
    # options = list(random.choice(list(audio_dataset['train'])).keys())
    # split1, split2 = random.sample(options, 2)
    # choice1, choice2 = (row[split1], row[split2])
    # if chosenmodel1 in model_names:
    #     chosenmodel1 = model_names[chosenmodel1]
    # if chosenmodel2 in model_names:
    #     chosenmodel2 = model_names[chosenmodel2]
    # out = [
    #     (choice1['sampling_rate'], choice1['array']),
    #     (choice2['sampling_rate'], choice2['array']),
    #     split1,
    #     split2
    # ]
    # if userid: out.append(userid)
    # if chosenmodel1: out.append(f'This model was {chosenmodel1}')
    # if chosenmodel2: out.append(f'This model was {chosenmodel2}')
    # return out
    # return (f'This model was {chosenmodel1}', f'This model was {chosenmodel2}', gr.update(visible=False), gr.update(visible=False))
    # return (gr.update(variant='secondary', value=chosenmodel1, interactive=False), gr.update(variant='secondary', value=chosenmodel2, interactive=False))
    out = [
        gr.update(interactive=False, visible=False),
        gr.update(interactive=False, visible=False)
    ]
    if chose_a == True:
        out.append(gr.update(value=f'✨ {chosenmodel1}', interactive=False, visible=True))
        out.append(gr.update(value=f'🥹 {chosenmodel2}', interactive=False, visible=True))
    else:
        out.append(gr.update(value=f'🥹 {chosenmodel1}', interactive=False, visible=True))
        out.append(gr.update(value=f'✨ {chosenmodel1}', interactive=False, visible=True))
    return out

with gr.Blocks() as leaderboard:
    gr.Markdown(LDESC)
    # df = gr.Dataframe(interactive=False, value=get_leaderboard())
    df = gr.Dataframe(interactive=False, min_width=0, wrap=True, column_widths=[30, 200, 50, 75, 50])
    reloadbtn = gr.Button("Refresh")
    leaderboard.load(get_leaderboard, outputs=[df])
    reloadbtn.click(get_leaderboard, outputs=[df])
    gr.Markdown("DISCLAIMER: The licenses listed may not be accurate or up to date, you are responsible for checking the licenses before using the models. Also note that some models may have additional usage restrictions.")

# with gr.Blocks() as vote:
#     useridstate = gr.State()
#     gr.Markdown(INSTR)
#     # gr.LoginButton()
#     with gr.Row():
#         gr.HTML('<div align="left"><h3>Model A</h3></div>')
#         gr.HTML('<div align="right"><h3>Model B</h3></div>')
#     model1 = gr.Textbox(interactive=False, visible=False, lines=1, max_lines=1)
#     model2 = gr.Textbox(interactive=False, visible=False, lines=1, max_lines=1)
#     # with gr.Group():
#     #     with gr.Row():
#     #         prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A")
#     #         prevmodel2 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model B", text_align="right")
#     #     with gr.Row():
#     #         aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
#     #         aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
#     with gr.Group():
#         with gr.Row():
#             with gr.Column():
#                 with gr.Group():
#                     prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A", lines=1, max_lines=1)
#                     aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
#             with gr.Column():
#                 with gr.Group():
#                     prevmodel2 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model B", text_align="right", lines=1, max_lines=1)
#                     aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})


#     with gr.Row():
#         abetter = gr.Button("A is Better", variant='primary', scale=4)
#         # skipbtn = gr.Button("Skip", scale=1)
#         bbetter = gr.Button("B is Better", variant='primary', scale=4)
#     with gr.Row():
#         bothbad = gr.Button("Both are Bad", scale=2)
#         skipbtn = gr.Button("Skip", scale=1)
#         bothgood = gr.Button("Both are Good", scale=2)
#     outputs = [aud1, aud2, model1, model2, useridstate, prevmodel1, prevmodel2]
#     abetter.click(a_is_better, outputs=outputs, inputs=[model1, model2, useridstate])
#     bbetter.click(b_is_better, outputs=outputs, inputs=[model1, model2, useridstate])
#     skipbtn.click(b_is_better, outputs=outputs, inputs=[model1, model2, useridstate])

#     bothbad.click(both_bad, outputs=outputs, inputs=[model1, model2, useridstate])
#     bothgood.click(both_good, outputs=outputs, inputs=[model1, model2, useridstate])

#     vote.load(reload, outputs=[aud1, aud2, model1, model2])
def synthandreturn(text):
    text = text.strip()
    if len(text) > MAX_SAMPLE_TXT_LENGTH:
        raise gr.Error(f'You exceeded the limit of {MAX_SAMPLE_TXT_LENGTH} characters')
    if not text:
        raise gr.Error(f'You did not enter any text')
    # Get two random models
    mdl1, mdl2 = random.sample(list(AVAILABLE_MODELS.keys()), 2)
    return (
        text,
        "Synthesize",
        gr.update(visible=True), # r2
        mdl1, # model1
        mdl2, # model2
        # 'Vote to reveal model A', # prevmodel1
        router.predict(
            text,
            AVAILABLE_MODELS[mdl1],
            api_name="/synthesize"
        ), # aud1
        # 'Vote to reveal model B', # prevmodel2
        router.predict(
            text,
            AVAILABLE_MODELS[mdl2],
            api_name="/synthesize"
        ), # aud2
        gr.update(visible=True, interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )
with gr.Blocks() as vote:
    useridstate = gr.State()
    gr.Markdown(INSTR)
    with gr.Group():
        text = gr.Textbox(label="Enter text to synthesize", info="By entering text, you certify that it is either in the public domain or, if you are its author, you dedicate it into the public domain. You also must agree to the privacy statement in the About page.")
        btn = gr.Button("Synthesize", variant='primary')
    model1 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=False)
    model2 = gr.Textbox(interactive=False, lines=1, max_lines=1, visible=False)
    with gr.Row(visible=False) as r2:
        with gr.Column():
            with gr.Group():
                aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
                abetter = gr.Button("A is better", variant='primary')
                prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A", text_align="center", lines=1, max_lines=1, visible=False)
        with gr.Column():
            with gr.Group():
                aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
                bbetter = gr.Button("B is better", variant='primary')
                prevmodel2 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model B", text_align="center", lines=1, max_lines=1, visible=False)
    # outputs = [text, btn, r2, model1, model2, prevmodel1, aud1, prevmodel2, aud2, abetter, bbetter]
    outputs = [text, btn, r2, model1, model2, aud1, aud2, abetter, bbetter, prevmodel1, prevmodel2]
    btn.click(synthandreturn, inputs=[text], outputs=outputs)

    # nxt_outputs = [prevmodel1, prevmodel2, abetter, bbetter]
    nxt_outputs = [abetter, bbetter, prevmodel1, prevmodel2]
    abetter.click(a_is_better, outputs=nxt_outputs, inputs=[model1, model2, useridstate])
    bbetter.click(b_is_better, outputs=nxt_outputs, inputs=[model1, model2, useridstate])
    # skipbtn.click(b_is_better, outputs=outputs, inputs=[model1, model2, useridstate])

    # bothbad.click(both_bad, outputs=outputs, inputs=[model1, model2, useridstate])
    # bothgood.click(both_good, outputs=outputs, inputs=[model1, model2, useridstate])

    # vote.load(reload, outputs=[aud1, aud2, model1, model2])

with gr.Blocks() as about:
    gr.Markdown(ABOUT)
with gr.Blocks() as admin:
    rdb = gr.Button("Reload Audio Dataset")
    # rdb.click(reload_audio_dataset, outputs=rdb)
    with gr.Group():
        dbtext = gr.Textbox(label="Type \"delete db\" to confirm", placeholder="delete db")
        ddb = gr.Button("Delete DB")
    ddb.click(del_db, inputs=dbtext, outputs=ddb)
with gr.Blocks(theme=theme, css="footer {visibility: hidden}textbox{resize:none}", title="TTS Leaderboard") as demo:
    gr.Markdown(DESCR)
    gr.TabbedInterface([vote, leaderboard, about, admin], ['Vote', 'Leaderboard', 'About', 'Admin (ONLY IN BETA)'])
    if CITATION_TEXT:
        with gr.Row():
            with gr.Accordion("📙 Citation", open=False):
                gr.Markdown(f"If you use this data in your publication, please cite us!\n\nCopy the BibTeX citation to cite this source:\n\n```bibtext\n{CITATION_TEXT}\n```")


demo.queue(api_open=False).launch(show_api=False)