import gradio as gr
import pandas as pd
from langdetect import detect
from datasets import load_dataset
import threading, time, uuid, sqlite3, shutil, os, random, asyncio, threading
from pathlib import Path
from huggingface_hub import CommitScheduler, delete_file, hf_hub_download
from gradio_client import Client
import pyloudnorm as pyln
import soundfile as sf
import librosa
from detoxify import Detoxify
import os
import tempfile
from pydub import AudioSegment

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# from gradio_space_ci import enable_space_ci

# enable_space_ci()



toxicity = Detoxify('original')
with open('harvard_sentences.txt') as f:
    sents = f.read().strip().splitlines()
####################################
# Constants
####################################
AVAILABLE_MODELS = {
    'XTTSv2': 'xtts',
    # 'WhisperSpeech': 'whisperspeech',
    'ElevenLabs': 'eleven',
    # 'OpenVoice': 'openvoice',
    'OpenVoice V2': 'openvoicev2',
    'Play.HT 2.0': 'playht',
#    'MetaVoice': 'metavoice',
    'MeloTTS': 'melo',
    'StyleTTS 2': 'styletts2',
    'GPT-SoVITS': 'sovits',
    # 'Vokan TTS': 'vokan',
    'VoiceCraft 2.0': 'voicecraft',
    'Parler TTS': 'parler'
}

SPACE_ID = os.getenv('SPACE_ID')
MAX_SAMPLE_TXT_LENGTH = 300
MIN_SAMPLE_TXT_LENGTH = 10
DB_DATASET_ID = os.getenv('DATASET_ID')
DB_NAME = "database.db"

# If /data available => means local storage is enabled => let's use it!
DB_PATH = f"/data/{DB_NAME}" if os.path.isdir("/data") else DB_NAME
print(f"Using {DB_PATH}")
# AUDIO_DATASET_ID = "ttseval/tts-arena-new"
CITATION_TEXT = """@misc{tts-arena,
	title        = {Text to Speech Arena},
	author       = {mrfakename and Srivastav, Vaibhav and Fourrier, Clémentine and Pouget, Lucain and Lacombe, Yoach and main and Gandhi, Sanchit},
	year         = 2024,
	publisher    = {Hugging Face},
	howpublished = "\\url{https://huggingface.co/spaces/TTS-AGI/TTS-Arena}"
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
            vote INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS votelog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            chosen TEXT,
            rejected TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS spokentext (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spokentext TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
def get_db():
    return sqlite3.connect(DB_PATH)



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
router = Client("TTS-AGI/tts-router", hf_token=os.getenv('HF_TOKEN'))
####################################
# Gradio app
####################################
MUST_BE_LOGGEDIN = "Please login with Hugging Face to participate in the TTS Arena."
DESCR = """
# TTS Arena: Benchmarking TTS Models in the Wild

Vote to help the community find the best available text-to-speech model!
""".strip()
# INSTR = """
# ## Instructions

# * Listen to two anonymous models
# * Vote on which synthesized audio sounds more natural to you
# * If there's a tie, click Skip

# **When you're ready to begin, login and begin voting!** The model names will be revealed once you vote.
# """.strip()
INSTR = """
## 🗳️ Vote

* Input text (English only) to synthesize audio (or press 🎲 for random text).
* Listen to the two audio clips, one after the other.
* Vote on which audio sounds more natural to you.
* _Note: Model names are revealed after the vote is cast._

Note: It may take up to 30 seconds to synthesize audio.
""".strip()
request = ''
if SPACE_ID:
    request = f"""
### Request a model

Please [create a Discussion](https://huggingface.co/spaces/{SPACE_ID}/discussions/new) to request a model.
"""
ABOUT = f"""
## 📄 About

The TTS Arena evaluates leading speech synthesis models. It is inspired by LMsys's [Chatbot Arena](https://chat.lmsys.org/).

### Motivation

The field of speech synthesis has long lacked an accurate method to measure the quality of different models. Objective metrics like WER (word error rate) are unreliable measures of model quality, and subjective measures such as MOS (mean opinion score) are typically small-scale experiments conducted with few listeners. As a result, these measurements are generally not useful for comparing two models of roughly similar quality. To address these drawbacks, we are inviting the community to rank models in an easy-to-use interface, and opening it up to the public in order to make both the opportunity to rank models, as well as the results, more easily accessible to everyone.

### The Arena

The leaderboard allows a user to enter text, which will be synthesized by two models. After listening to each sample, the user can vote on which model sounds more natural. Due to the risks of human bias and abuse, model names are revealed only after a vote is submitted.

### Credits

Thank you to the following individuals who helped make this project possible:

* VB ([Twitter](https://twitter.com/reach_vb) / [Hugging Face](https://huggingface.co/reach-vb))
* Clémentine Fourrier ([Twitter](https://twitter.com/clefourrier) / [Hugging Face](https://huggingface.co/clefourrier))
* Lucain Pouget ([Twitter](https://twitter.com/Wauplin) / [Hugging Face](https://huggingface.co/Wauplin))
* Yoach Lacombe ([Twitter](https://twitter.com/yoachlacombe) / [Hugging Face](https://huggingface.co/ylacombe))
* Main Horse ([Twitter](https://twitter.com/main_horse) / [Hugging Face](https://huggingface.co/main-horse))
* Sanchit Gandhi ([Twitter](https://twitter.com/sanchitgandhi99) / [Hugging Face](https://huggingface.co/sanchit-gandhi))
* Apolinário Passos ([Twitter](https://twitter.com/multimodalart) / [Hugging Face](https://huggingface.co/multimodalart))
* Pedro Cuenca ([Twitter](https://twitter.com/pcuenq) / [Hugging Face](https://huggingface.co/pcuenq))

{request}

### Privacy statement

We may store text you enter and generated audio. We store a unique ID for each session. You agree that we may collect, share, and/or publish any data you input for research and/or commercial purposes.

### License

Generated audio clips cannot be redistributed and may be used for personal, non-commercial use only.

Random sentences are sourced from a filtered subset of the [Harvard Sentences](https://www.cs.columbia.edu/~hgs/audio/harvard.html).
""".strip()
LDESC = """
## 🏆 Leaderboard

Vote to help the community determine the best text-to-speech (TTS) models.

The leaderboard displays models in descending order of how natural they sound (based on votes cast by the community).

Important: In order to help keep results fair, the leaderboard hides results by default until the number of votes passes a threshold. Tick the `Reveal preliminary results` to show models without sufficient votes. Please note that preliminary results may be inaccurate.
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
    'eleven': 'Proprietary',
    'openai': 'Proprietary',
    'hierspeech': 'MIT',
    'pheme': 'CC-BY',
    'speecht5': 'MIT',
    'metavoice': 'Apache 2.0',
    'elevenlabs': 'Proprietary',
    'whisperspeech': 'MIT',
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
def model_license(name):
    print(name)
    for k, v in AVAILABLE_MODELS.items():
        if k == name:
            if v in model_licenses:
                return model_licenses[v]
    print('---')
    return 'Unknown'
def get_leaderboard(reveal_prelim = False):
    conn = get_db()
    cursor = conn.cursor()
    sql = 'SELECT name, upvote, downvote FROM model'
    # if not reveal_prelim: sql += ' WHERE EXISTS (SELECT 1 FROM model WHERE (upvote + downvote) > 750)'
    if not reveal_prelim: sql += ' WHERE (upvote + downvote) > 500'
    cursor.execute(sql)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['name', 'upvote', 'downvote'])
    # df['license'] = df['name'].map(model_license)
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
    # df = df[['order', 'name', 'score', 'license', 'votes']]
    df = df[['order', 'name', 'score', 'votes']]
    return df
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
def log_text(text):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO spokentext (spokentext) VALUES (?)', (text,))
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
    print("A is better", model1, model2)
    if not model1 in AVAILABLE_MODELS.keys() and not model1 in AVAILABLE_MODELS.values():
        raise gr.Error('Sorry, please try voting again.')
    userid = mkuuid(userid)
    if model1 and model2:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO votelog (username, chosen, rejected) VALUES (?, ?, ?)', (str(userid), model1, model2,))
        with scheduler.lock:
            conn.commit()
            cursor.close()
        upvote_model(model1, str(userid))
        downvote_model(model2, str(userid))
    return reload(model1, model2, userid, chose_a=True)
def b_is_better(model1, model2, userid):
    print("B is better", model1, model2)
    if not model1 in AVAILABLE_MODELS.keys() and not model1 in AVAILABLE_MODELS.values():
        raise gr.Error('Sorry, please try voting again.')
    userid = mkuuid(userid)
    if model1 and model2:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO votelog (username, chosen, rejected) VALUES (?, ?, ?)', (str(userid), model2, model1,))
        with scheduler.lock:
            conn.commit()
            cursor.close()
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
        out.append(gr.update(value=f'Your vote: {chosenmodel1}', interactive=False, visible=True))
        out.append(gr.update(value=f'{chosenmodel2}', interactive=False, visible=True))
    else:
        out.append(gr.update(value=f'{chosenmodel1}', interactive=False, visible=True))
        out.append(gr.update(value=f'Your vote: {chosenmodel2}', interactive=False, visible=True))
    out.append(gr.update(visible=True))
    return out

with gr.Blocks() as leaderboard:
    gr.Markdown(LDESC)
    # df = gr.Dataframe(interactive=False, value=get_leaderboard())
    df = gr.Dataframe(interactive=False, min_width=0, wrap=True, column_widths=[30, 200, 50, 50])
    with gr.Row():
        reveal_prelim = gr.Checkbox(label="Reveal preliminary results", info="Show all models, including models with very few human ratings.", scale=1)
        reloadbtn = gr.Button("Refresh", scale=3)
    reveal_prelim.input(get_leaderboard, inputs=[reveal_prelim], outputs=[df])
    leaderboard.load(get_leaderboard, inputs=[reveal_prelim], outputs=[df])
    reloadbtn.click(get_leaderboard, inputs=[reveal_prelim], outputs=[df])
    # gr.Markdown("DISCLAIMER: The licenses listed may not be accurate or up to date, you are responsible for checking the licenses before using the models. Also note that some models may have additional usage restrictions.")

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
def doloudnorm(path):
    data, rate = sf.read(path)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)
    sf.write(path, loudness_normalized_audio, rate)

def doresample(path_to_wav):
    pass
##########################
# 2x speedup (hopefully) #
##########################

def synthandreturn(text):
    text = text.strip()
    if len(text) > MAX_SAMPLE_TXT_LENGTH:
        raise gr.Error(f'You exceeded the limit of {MAX_SAMPLE_TXT_LENGTH} characters')
    if len(text) < MIN_SAMPLE_TXT_LENGTH:
        raise gr.Error(f'Please input a text longer than {MIN_SAMPLE_TXT_LENGTH} characters')
    if (
        # test toxicity if not prepared text
        text not in sents
        and toxicity.predict(text)['toxicity'] > 0.8
    ):
        print(f'Detected toxic content! "{text}"')
        raise gr.Error('Your text failed the toxicity test')
    if not text:
        raise gr.Error(f'You did not enter any text')
    # Check language
    try:
        if not detect(text) == "en":
            gr.Warning('Warning: The input text may not be in English')
    except:
        pass
    # Get two random models
    mdl1, mdl2 = random.sample(list(AVAILABLE_MODELS.keys()), 2)
    log_text(text)
    print("[debug] Using", mdl1, mdl2)
    def predict_and_update_result(text, model, result_storage):
        try:
            if model in AVAILABLE_MODELS:
                result = router.predict(text, AVAILABLE_MODELS[model].lower(), api_name="/synthesize")
            else:
                result = router.predict(text, model.lower(), api_name="/synthesize")
        except:
            raise gr.Error('Unable to call API, please try again :)')
        print('Done with', model)
        # try:
        #     doresample(result)
        # except:
        #     pass
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                audio = AudioSegment.from_file(result)
                current_sr = audio.frame_rate
                if current_sr > 24000:
                    audio = audio.set_frame_rate(24000)
                try:
                    print('Trying to normalize audio')
                    audio = match_target_amplitude(audio, -20)
                except:
                    print('[WARN] Unable to normalize audio')
                audio.export(f.name, format="wav")
                os.unlink(result)
                result = f.name
        except:
            pass
        if model in AVAILABLE_MODELS.keys(): model = AVAILABLE_MODELS[model]
        print(model)
        print(f"Running model {model}")
        result_storage[model] = result
        # try:
        #     doloudnorm(result)
        # except:
        #     pass
    mdl1k = mdl1
    mdl2k = mdl2
    print(mdl1k, mdl2k)
    if mdl1 in AVAILABLE_MODELS.keys(): mdl1k=AVAILABLE_MODELS[mdl1]
    if mdl2 in AVAILABLE_MODELS.keys(): mdl2k=AVAILABLE_MODELS[mdl2]
    results = {}
    print(f"Sending models {mdl1k} and {mdl2k} to API")
    thread1 = threading.Thread(target=predict_and_update_result, args=(text, mdl1k, results))
    thread2 = threading.Thread(target=predict_and_update_result, args=(text, mdl2k, results))
    
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    #debug
    # print(results)
    # print(list(results.keys())[0])
    # y, sr = librosa.load(results[list(results.keys())[0]], sr=None)
    # print(sr)
    # print(list(results.keys())[1])
    # y, sr = librosa.load(results[list(results.keys())[1]], sr=None)
    # print(sr)
    #debug
    #     outputs = [text, btn, r2, model1, model2, aud1, aud2, abetter, bbetter, prevmodel1, prevmodel2, nxtroundbtn]
    
    print(f"Retrieving models {mdl1k} and {mdl2k} from API")
    return (
        text,
        "Synthesize",
        gr.update(visible=True), # r2
        mdl1, # model1
        mdl2, # model2
        gr.update(visible=True, value=results[mdl1k]), # aud1
        gr.update(visible=True, value=results[mdl2k]), # aud2
        gr.update(visible=True, interactive=False), #abetter
        gr.update(visible=True, interactive=False), #bbetter
        gr.update(visible=False), #prevmodel1
        gr.update(visible=False), #prevmodel2
        gr.update(visible=False), #nxt round btn
    )
    # return (
    #     text,
    #     "Synthesize",
    #     gr.update(visible=True), # r2
    #     mdl1, # model1
    #     mdl2, # model2
    #     # 'Vote to reveal model A', # prevmodel1
    #     gr.update(visible=True, value=router.predict(
    #         text,
    #         AVAILABLE_MODELS[mdl1],
    #         api_name="/synthesize"
    #     )), # aud1
    #     # 'Vote to reveal model B', # prevmodel2
    #     gr.update(visible=True, value=router.predict(
    #         text,
    #         AVAILABLE_MODELS[mdl2],
    #         api_name="/synthesize"
    #     )), # aud2
    #     gr.update(visible=True, interactive=True),
    #     gr.update(visible=True, interactive=True),
    #     gr.update(visible=False),
    #     gr.update(visible=False),
    #     gr.update(visible=False), #nxt round btn
    # )

def unlock_vote(btn_index, aplayed, bplayed):
    # sample played
    if btn_index == 0:
        aplayed = gr.State(value=True)
    if btn_index == 1:
        bplayed = gr.State(value=True)

    # both audio samples played
    if bool(aplayed) and bool(bplayed):
        print('Both audio samples played, voting unlocked')
        return [gr.update(interactive=True), gr.update(interactive=True), gr.update(), gr.update()]

    return [gr.update(), gr.update(), aplayed, bplayed]

def randomsent():
    return random.choice(sents), '🎲'
def clear_stuff():
    return "", "Synthesize", gr.update(visible=False), '', '', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

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
                aud1 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
                abetter = gr.Button("A is better", variant='primary')
                prevmodel1 = gr.Textbox(interactive=False, show_label=False, container=False, value="Vote to reveal model A", text_align="center", lines=1, max_lines=1, visible=False)
        with gr.Column():
            with gr.Group():
                aud2 = gr.Audio(interactive=False, show_label=False, show_download_button=False, show_share_button=False, waveform_options={'waveform_progress_color': '#3C82F6'})
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
    # skipbtn.click(b_is_better, outputs=outputs, inputs=[model1, model2, useridstate])

    # bothbad.click(both_bad, outputs=outputs, inputs=[model1, model2, useridstate])
    # bothgood.click(both_good, outputs=outputs, inputs=[model1, model2, useridstate])

    # vote.load(reload, outputs=[aud1, aud2, model1, model2])

with gr.Blocks() as about:
    gr.Markdown(ABOUT)
# with gr.Blocks() as admin:
#     rdb = gr.Button("Reload Audio Dataset")
#     # rdb.click(reload_audio_dataset, outputs=rdb)
#     with gr.Group():
#         dbtext = gr.Textbox(label="Type \"delete db\" to confirm", placeholder="delete db")
#         ddb = gr.Button("Delete DB")
#     ddb.click(del_db, inputs=dbtext, outputs=ddb)
with gr.Blocks(theme=theme, css="footer {visibility: hidden}textbox{resize:none}", title="TTS Arena") as demo:
    gr.Markdown(DESCR)
    # gr.TabbedInterface([vote, leaderboard, about, admin], ['Vote', 'Leaderboard', 'About', 'Admin (ONLY IN BETA)'])
    gr.TabbedInterface([vote, leaderboard, about], ['🗳️ Vote', '🏆 Leaderboard', '📄 About'])
    if CITATION_TEXT:
        with gr.Row():
            with gr.Accordion("Citation", open=False):
                gr.Markdown(f"If you use this data in your publication, please cite us!\n\nCopy the BibTeX citation to cite this source:\n\n```bibtext\n{CITATION_TEXT}\n```\n\nPlease remember that all generated audio clips should be assumed unsuitable for redistribution or commercial use.")


demo.queue(api_open=False, default_concurrency_limit=40).launch(show_api=False)