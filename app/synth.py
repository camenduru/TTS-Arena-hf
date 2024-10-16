from .models import *
from .utils import *
from .config import *
from .init import *

import gradio as gr
from pydub import AudioSegment
import random, os, threading, tempfile
from langdetect import detect
from .vote import log_text

def random_m():
    return random.sample(list(set(AVAILABLE_MODELS.keys())), 2)

def check_toxicity(text):
    if not TOXICITY_CHECK:
        return False
    return toxicity.predict(text)['toxicity'] > 0.8

def synthandreturn(text):
    text = text.strip()
    if len(text) > MAX_SAMPLE_TXT_LENGTH:
        raise gr.Error(f'You exceeded the limit of {MAX_SAMPLE_TXT_LENGTH} characters')
    if len(text) < MIN_SAMPLE_TXT_LENGTH:
        raise gr.Error(f'Please input a text longer than {MIN_SAMPLE_TXT_LENGTH} characters')
    if (
        # test toxicity if not prepared text
        text not in sents
        and check_toxicity(text)
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

# Battle Mode

def synthandreturn_battle(text, mdl1, mdl2):
    if mdl1 == mdl2:
        raise gr.Error('You can\'t pick two of the same models.')
    text = text.strip()
    if len(text) > MAX_SAMPLE_TXT_LENGTH:
        raise gr.Error(f'You exceeded the limit of {MAX_SAMPLE_TXT_LENGTH} characters')
    if len(text) < MIN_SAMPLE_TXT_LENGTH:
        raise gr.Error(f'Please input a text longer than {MIN_SAMPLE_TXT_LENGTH} characters')
    if (
        # test toxicity if not prepared text
        text not in sents
        and check_toxicity(text)
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

# Unlock vote

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
def randomsent_battle():
    return tuple(randomsent()) + tuple(random_m())
def clear_stuff():
    return "", "Synthesize", gr.update(visible=False), '', '', gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)