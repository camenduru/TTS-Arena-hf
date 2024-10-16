import uuid
import soundfile as sf
import pydub
import pyloudnorm as pyln

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def mkuuid(uid):
    if not uid:
        uid = uuid.uuid4()
    return uid

def doloudnorm(path):
    data, rate = sf.read(path)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)
    sf.write(path, loudness_normalized_audio, rate)