from .config import *
from .db import *
from huggingface_hub import CommitScheduler
from pathlib import Path
from gradio_client import Client
import os


scheduler = None

if SYNC_DB:
    download_db()
    # Sync local DB with remote repo every 5 minute (only if a change is detected)
    scheduler = CommitScheduler(
        repo_id=DB_DATASET_ID,
        repo_type="dataset",
        folder_path=Path(DB_PATH).parent,
        every=5,
        allow_patterns=DB_NAME,
    )

create_db()

# Load TTS Router
router = Client(ROUTER_ID, hf_token=os.getenv('HF_TOKEN'))

if TOXICITY_CHECK:
    # Load toxicity model
    from detoxify import Detoxify
    toxicity = Detoxify('original')