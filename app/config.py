import os

# NOTE: Configure models in `models.py`

#########################
# General Configuration #
#########################

DB_NAME = "database.db"

TOXICITY_CHECK = False

MAX_SAMPLE_TXT_LENGTH = 300 # Maximum text length (characters)
MIN_SAMPLE_TXT_LENGTH = 10 # Minimum text length (characters)

DB_PATH = f"/data/{DB_NAME}" if os.path.isdir("/data") else DB_NAME # If /data available => means local storage is enabled => let's use it!

ROUTER_ID = "TTS-AGI/tts-router" # You should use a router space to route TTS models to avoid exposing your API keys!

SYNC_DB = True # Sync DB to HF dataset?
DB_DATASET_ID = os.getenv('DATASET_ID') # HF dataset ID, can be None if not syncing

SPACE_ID = os.getenv('SPACE_ID') # Don't change this! It detects if we're running in a HF Space

with open(os.path.dirname(__file__) + '/../harvard_sentences.txt', 'r') as f:
    sents = f.read().strip().splitlines()

######################
# TTS Arena Settings #
######################

CITATION_TEXT = """@misc{tts-arena,
	title        = {Text to Speech Arena},
	author       = {mrfakename and Srivastav, Vaibhav and Fourrier, Clémentine and Pouget, Lucain and Lacombe, Yoach and main and Gandhi, Sanchit},
	year         = 2024,
	publisher    = {Hugging Face},
	howpublished = "\\url{https://huggingface.co/spaces/TTS-AGI/TTS-Arena}"
}"""