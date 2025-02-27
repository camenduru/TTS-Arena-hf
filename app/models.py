# Models to include in the leaderboard, only include models that users can vote on
AVAILABLE_MODELS = {
    'XTTSv2': 'xtts',
    # 'WhisperSpeech': 'whisperspeech',
    'ElevenLabs': 'eleven',
    # 'OpenVoice': 'openvoice',
    'OpenVoice V2': 'openvoicev2',
    'Play.HT 2.0': 'playht',
    'Play.HT 3.0 Mini': 'playht3',
    # 'MetaVoice': 'metavoice',
    'MeloTTS': 'melo',
    'StyleTTS 2': 'styletts2',
    'GPT-SoVITS': 'sovits',
    # 'Vokan TTS': 'vokan',
    'VoiceCraft 2.0': 'voicecraft',
    'Parler TTS': 'parler',
    'Parler TTS Large': 'parlerlarge',
    'Fish Speech v1.4': 'fish',
}


# Model name mapping, can include models that users cannot vote on
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