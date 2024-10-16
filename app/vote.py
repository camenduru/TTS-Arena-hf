from .utils import *
from .config import *
from .models import *
from .db import *
from .init import *

import gradio as gr

# Logging

def log_text(text):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO spokentext (spokentext) VALUES (?)', (text,))
    if scheduler:
        with scheduler.lock:
            conn.commit()
    else:
        conn.commit()
    cursor.close()

# Vote

def upvote_model(model, uname, battle=False):
    conn = get_db()
    cursor = conn.cursor()
    if battle: uname = "unknown_battle"
    cursor.execute('UPDATE model SET upvote = upvote + 1 WHERE name = ?', (model,))
    if cursor.rowcount == 0:
        cursor.execute('INSERT OR REPLACE INTO model (name, upvote, downvote) VALUES (?, 1, 0)', (model,))
    cursor.execute('INSERT INTO vote (username, model, vote) VALUES (?, ?, ?)', (uname, model, 1,))
    if scheduler:
        with scheduler.lock:
            conn.commit()
    else:
        conn.commit()
    cursor.close()

def downvote_model(model, uname, battle=False):
    conn = get_db()
    cursor = conn.cursor()
    if battle: uname = "unknown_battle"
    cursor.execute('UPDATE model SET downvote = downvote + 1 WHERE name = ?', (model,))
    if cursor.rowcount == 0:
        cursor.execute('INSERT OR REPLACE INTO model (name, upvote, downvote) VALUES (?, 0, 1)', (model,))
    cursor.execute('INSERT INTO vote (username, model, vote) VALUES (?, ?, ?)', (uname, model, -1,))
    if scheduler:
        with scheduler.lock:
            conn.commit()
    else:
        conn.commit()
    cursor.close()

# Battle Mode

def a_is_better_battle(model1, model2, userid):
    return a_is_better(model1, model2, 'unknown_battle', True)
def b_is_better_battle(model1, model2, userid):
    return b_is_better(model1, model2, 'unknown_battle', True)

# A/B better

def a_is_better(model1, model2, userid, battle=False):
    print("A is better", model1, model2)
    if not model1 in AVAILABLE_MODELS.keys() and not model1 in AVAILABLE_MODELS.values():
        raise gr.Error('Sorry, please try voting again.')
    userid = mkuuid(userid)
    if model1 and model2:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO votelog (username, chosen, rejected) VALUES (?, ?, ?)', (str(userid), model1, model2,))
        if scheduler:
            with scheduler.lock:
                conn.commit()
        else:
            conn.commit()
        cursor.close()
        upvote_model(model1, str(userid), battle)
        downvote_model(model2, str(userid), battle)
    return reload(model1, model2, userid, chose_a=True)
def b_is_better(model1, model2, userid, battle=False):
    print("B is better", model1, model2)
    if not model1 in AVAILABLE_MODELS.keys() and not model1 in AVAILABLE_MODELS.values():
        raise gr.Error('Sorry, please try voting again.')
    userid = mkuuid(userid)
    if model1 and model2:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO votelog (username, chosen, rejected) VALUES (?, ?, ?)', (str(userid), model2, model1,))
        if scheduler:
            with scheduler.lock:
                conn.commit()
        else:
            conn.commit()
        cursor.close()
        upvote_model(model2, str(userid), battle)
        downvote_model(model1, str(userid), battle)
    return reload(model1, model2, userid, chose_b=True)

# Reload

def reload(chosenmodel1=None, chosenmodel2=None, userid=None, chose_a=False, chose_b=False):
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