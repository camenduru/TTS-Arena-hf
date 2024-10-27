from .config import *
from .models import *

############
# Messages #
############

MUST_BE_LOGGEDIN = "Please login with Hugging Face to participate in the TTS Arena."
DESCR = """
# TTS Arena: Benchmarking TTS Models in the Wild
Vote to help the community find the best available text-to-speech model!
""".strip()
BATTLE_INSTR = """
## Battle
Choose 2 candidates and vote on which one is better! Currently in beta.
* Input text (English only) to synthesize audio (or press 🎲 for random text).
* Listen to the two audio clips, one after the other.
* Vote on which audio sounds more natural to you.
"""
INSTR = """
## Vote
* Input text (English only) to synthesize audio (or press 🎲 for random text).
* Listen to the two audio clips, one after the other.
* Vote on which audio sounds more natural to you.
* _Note: Model names are revealed after the vote is cast._
Note: It may take up to 30 seconds to synthesize audio.
""".strip()
request = ""
if SPACE_ID:
    request = f"""
### Request a model
Please [create a Discussion](https://huggingface.co/spaces/{SPACE_ID}/discussions/new) to request a model.
"""
ABOUT = f"""
## About
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


model_series = []
for model in AVAILABLE_MODELS.keys():
    # name up to first whitespace
    model = model.split()[0]
    model_series.append('%27'+ model +'%27')
try:
    for model in HF_SPACES.values():
        # url encode pluses +
        model_series.append('%27'+ model['series'].replace('+', '%2B') +'%27')
except:
    pass

TTS_INFO = f"""
## 🗣 Contenders

### 🔐 Closed Source TTS
* ElevenLabs
* Play.ht

### 🔓 Open Source TTS capabilities table

See [the full dataset itself](https://huggingface.co/datasets/Pendrokar/open_tts_tracker) for the legend and more in depth information of each model.
""".strip()
TTS_DATASET_IFRAME_ORDER = '%2C+'.join(model_series)
TTS_DATASET_IFRAME = f"""
<iframe
    src="https://huggingface.co/datasets/Pendrokar/open_tts_tracker/embed/sql-console/default/train?sql_console=true&sql=--+The+SQL+console+is+powered+by+DuckDB+WASM+and+runs+entirely+in+the+browser.%0A--+Get+started+by+typing+a+query+or+selecting+a+view+from+the+options+below.%0ASELECT+*%2C+%22Name%22+IN+%28{TTS_DATASET_IFRAME_ORDER}%29+AS+%22In+arena%22+FROM+train+WHERE+%22Insta-clone+%F0%9F%91%A5%22+IS+NOT+NULL+ORDER+BY+%22In+arena%22+DESC+LIMIT+50%3B&views%5B%5D=train"
    frameborder="0"
    width="100%"
    height="650px"
></iframe>
""".strip()