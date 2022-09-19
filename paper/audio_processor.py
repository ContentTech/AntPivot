import json
import random
import re
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt


def time2milsec(timef):
    timef = timef.split(":")
    timef = [x for x in timef if x and "0" <= x[0] <= "9"]
    assert len(timef) == 3
    h = int(timef[0])
    m = int(timef[1])
    if "." in timef[2]:
        s, ms = timef[2].split(".")
    else:
        s = timef[2]
        ms = 0
    s = int(s)
    ms = int(ms)
    return 3600000 * h + 60000 * m + 1000 * s + ms


def transform_time_string(time):
    start_str, end_str = time.split('-')
    start_time = time2milsec(start_str) / 1000.0
    end_time = time2milsec(end_str) / 1000.0
    return [start_time, end_time]

def get_sentence_label(text, labels, prelabel):
    for label in labels:
        if text in label:
            label_type = label[:label.find(";")]
            assert label_type in ['start', 'end'], 'label {} not in (start, end)'.format(label_type)
            if label_type == 'start':
                return 1
            else:
                return 3
    if prelabel == 0:
        return 0
    elif prelabel == 1:
        return 2
    elif prelabel == 2:
        return 2
    else:
        return 0

def capture_interval(wave_form, start_time, end_time, sample_rate):
    start_pos = int(start_time * sample_rate)
    end_pos = int(end_time * sample_rate)
    sub_wave = wave_form[:, start_pos:end_pos]
    return torchaudio.transforms.MelSpectrogram(n_fft=1024, n_mels=32)(sub_wave)

wav_name = "audio.wav"
asr_file = "asr.json"
label_file = "label.json"
waveform, sample_rate = torchaudio.load(wav_name)
print(waveform.size())


asr_text = json.load(open(asr_file))
labels = json.load(open(label_file))
num = len(asr_text)


pre_label = 0

audio_bin = {
    "0": [],
    "1": [],
    "2": [],
    "3": []
}

for sent_idx in range(num):
    sentence = asr_text[sent_idx]
    if len(sentence)==0:
        continue
    time, text = re.split(": spk., |: speak., ", sentence)
    start_time, end_time = transform_time_string(time)
    # get sentence embedding
    sent_label = get_sentence_label(text, labels, pre_label)
    pre_label = sent_label
    audio_bin[str(sent_label)].append(capture_interval(waveform, start_time, end_time, sample_rate))
#
type_dict = {
    "0": "start", "1": "inside", "2": "end", "3": "other"
}

for label_type in ["0", "1", "2", "3"]:
    current_bin = np.array(audio_bin[label_type])
    audio_num = len(current_bin)
    audio_idx = np.random.choice(audio_num, size=5, replace=False)
    sampled_audio = current_bin[audio_idx]
    for i, audio in enumerate(sampled_audio):
        pool = audio.log2()[0,:,:].transpose(-2, -1).mean(0, keepdim=True)
        plt.imshow(pool.detach().numpy())
        plt.savefig(type_dict[label_type] + str(i) + ".jpg")
