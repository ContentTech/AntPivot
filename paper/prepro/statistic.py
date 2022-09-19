# encoding=utf-8

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

import numpy as np
import sys
import json
import argparse
import re
import jieba
import logging
import torch
import torchaudio

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

INGORE_WORD = set(['，', '的'])
spk_map = {"speak0": 0,
           "spk1": 1,
           "spk2": 2
           }


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", default='data/raw')
    parser.add_argument('-dataset', default='train', help='train, valid or test ')
    parser.add_argument('-batch_size', default=100, type=int, help='save batch size')
    parser.add_argument("-save_dir", default='data/output')
    parser.add_argument("-device", default='cuda:0')

    parser.add_argument("-logs_file", default='logs/preprocess.log')
    args = parser.parse_args()
    return args


def get_sentence_speaker(sentence):
    sub_sentence = sentence[sentence.find('s'):]
    spk_info = sub_sentence[:sub_sentence.find(',')]
    logger.debug('speaker info: {}'.format(spk_info))
    return spk_map.get(spk_info, 0)


def capture_interval(wave_form, start_time, end_time, sample_rate):
    max_pos = wave_form.size(-1)
    start_pos = int(start_time * sample_rate)
    end_pos = int(end_time * sample_rate)
    sub_wave = wave_form[:, start_pos:end_pos]
    if start_pos > max_pos or end_pos > max_pos:
        return np.zeros((8, 128))
    sub_mel = torchaudio.transforms.MelSpectrogram(n_fft=1024)(sub_wave)  # [1, 128, width]
    pooled_mel = torch.adaptive_avg_pool1d(sub_mel, output_size=8).squeeze(0).transpose(-2, -1).detach().cpu().numpy()
    # [8, 128]
    return pooled_mel


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

def get_sentence_length(text):
    return len([word for word in jieba.lcut(text) if word not in INGORE_WORD])


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

def analyze_asr(asr_list):
    first_start, last_end = float('inf'), -float('inf')
    all_interval = []
    all_length = []
    for raw_sent in asr_list:
        if len(raw_sent) == 0:
            continue
        time, text = re.split(": spk., |: speak., ", raw_sent)
        all_length.append(get_sentence_length(text))
        start_time, end_time = transform_time_string(time)
        first_start = min(first_start, start_time)
        last_end = max(last_end, end_time)
        all_interval.append(max(end_time - start_time, 0))
    return last_end - first_start, all_interval, all_length
    # float, List[float], List[int]

def analyze_label(label_list):
    all_interval = []
    all_length = []
    for sent in label_list:
        if len(sent) == 0:
            continue
        label, sent = re.split(";", sent)
        time, text = re.split(": spk., |: speak., ", sent)
        all_length.append(get_sentence_length(text))
        start_time, end_time = transform_time_string(time)
        all_interval.append(max(end_time - start_time, 0))
    return all_interval, all_length
    # List[float], List[int]

def calc_utterance_per_highlight(asr, label):
    pre_label = 0
    counter = 0
    inside = False
    all_labels = []
    all_utter = []
    for sentence in asr:
        if len(sentence) == 0:
            continue
        time, text = re.split(": spk., |: speak., ", sentence)
        sent_label = get_sentence_label(text, label, pre_label)
        all_labels.append(sent_label)
        pre_label = sent_label
    for label in all_labels:
        if label == 1 and not inside:
            inside, counter = True, 0
        if label == 2 and inside:
            counter += 1
        if label == 3 and inside:
            inside = False
            all_utter.append(counter)
    return all_utter

def statistic():
    ignored_list = ["1081381d74a54ca1874a3be124f50a92_s_1.json", ]
    args = arg_parse()
    logger.info(args)
    all_sent_num = []
    all_entire_time = []
    all_split_num = []
    all_word_num = []
    split_word_num = []
    all_talk_time = []
    all_highlight_time = []
    all_sent_time = []
    talk_ratio = []
    highlight_ratio = []
    utterance_per_highlight = []
    for dataset in ["train", "test", "eval"]:
        data_dir = os.path.join(args.data_dir, dataset)
        file_name_list = os.listdir(os.path.join(data_dir, 'asr'))
        for idx, name in tqdm(enumerate(file_name_list)):
            if name in ignored_list:
                continue
            logger.info('process file {}'.format(name))
            asr_file = os.path.join(data_dir, 'asr', name)
            label_file = os.path.join(data_dir, 'label', name)
            if not os.path.exists(label_file):
                continue

            asr_text = json.load(open(asr_file))
            label_text = json.load(open(label_file))
            utterance_per_highlight.extend(calc_utterance_per_highlight(asr_text, label_text))

            all_sent_num.append(len(asr_text))
            all_split_num.append(len(label_text) // 2)
            entire_time, talk_time_list, entire_word_num = analyze_asr(asr_text)
            # float, List[float], List[int]
            all_entire_time.append(entire_time)
            if entire_time < 200 or all_split_num == 0:
                print(name)
                sys.stdout.flush()
            all_talk_time.append(sum(talk_time_list))
            all_sent_time.extend(talk_time_list)
            all_word_num.extend(entire_word_num)
            highlight_time, highlight_word_num = analyze_label(label_text)
            # List[float], List[int]
            highlight_time_ratio = sum(highlight_time) / sum(talk_time_list)
            talk_time_ratio = sum(talk_time_list) / entire_time
            highlight_ratio.append(highlight_time_ratio)
            talk_ratio.append(talk_time_ratio)

            all_highlight_time.append(sum(highlight_time))
            split_word_num.extend(highlight_word_num)
    return all_entire_time, all_talk_time, all_highlight_time, all_sent_num, all_split_num, \
           all_word_num, split_word_num, all_sent_time, highlight_ratio, talk_ratio, utterance_per_highlight

def mean(seq):
    return 1.0 * sum(seq) / len(seq)



if __name__ == '__main__':
    entire_time, talk_time, highlight_time, sent_num, highlight_num, all_word_num, \
    highlight_word_num, all_sent_time, highlight_ratio, talk_ratio, uph = statistic()
    results = {
        "total_duration": entire_time,
        "highlight_ratio": highlight_ratio,
        "highlight_time": highlight_time,
        "highlight_num": highlight_num,
        "utterance_ratio": talk_ratio,
        "utterance_time": all_sent_time,
        "utterance_num": sent_num,
        "word_per_utterance": all_word_num,
        "utterance_per_highlight": uph
    }
    with open("statistic.pkl", "wb") as f:
        pickle.dump(results, f)
    # print("Sum. Entire Time:", sum(entire_time))
    # print("Avg. Entire Time: ", mean(entire_time))
    # print("Avg. Talk Time: ", mean(talk_time))
    # print("Avg. Highlight Time: ", mean(highlight_time))
    # print("Avg. Sent Time: ", mean(all_sent_time))
    # print("Avg. Sent Num: ", mean(sent_num))
    # print("Avg. Highlight Num: ", mean(highlight_num))
    # print("Avg. Sent Len: ", mean(all_word_num))
    # print("Avg. Highlight Sent Len: ", mean(highlight_word_num))
    # print("Avg. Talk Ratio: ", mean(talk_ratio))
    # print("Avg. Highlight Ratio:", mean(highlight_ratio))
    # print("Avg. Utter per highlight:", mean(uph))
