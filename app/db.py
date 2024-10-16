import sqlite3
from .config import *
import os
import shutil
from huggingface_hub import hf_hub_download

def download_db():
    if not os.path.isfile(DB_PATH):
        print("Downloading DB...")
        try:
            cache_path = hf_hub_download(repo_id=DB_DATASET_ID, repo_type='dataset', filename=DB_NAME)
            shutil.copyfile(cache_path, DB_PATH)
            print("Downloaded DB")
        except Exception as e:
            print("Error while downloading DB:", e)

def get_db():
    return sqlite3.connect(DB_PATH)

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