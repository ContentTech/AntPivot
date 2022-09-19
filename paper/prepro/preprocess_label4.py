# encoding=utf-8

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    parser.add_argument("-data_dir", default='./data')
    parser.add_argument('-dataset', default='test', help='train, eval or test ')
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


def get_sentence_embedding(text, sent_idx, model):
    new_text = ''.join([word for word in jieba.lcut(text) if word not in INGORE_WORD])
    embeddings = model.encode([new_text], batch_size=1, show_progress_bar=False)
    # logger.debug('embeding data:')
    # logger.debug(embeddings)
    return embeddings[0]

def get_sentence_length(text):
    return len([word for word in jieba.lcut(text) if word not in INGORE_WORD])

def judge_label_pairs(labels):
    num = len(labels)
    if num % 2 != 0:
        return False
    for i in range(num//2):
        if 'start' not in labels[2*i] or 'end' not in labels[2*i+1]:
           return False
    return True

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


def get_docs_embedding(name, asr_file, label_file, audio_file, model=None):
    asr_text = json.load(open(asr_file))
    labels = json.load(open(label_file))
    waveform, sample_rate = torchaudio.load(audio_file)
    
    if not judge_label_pairs(labels):
        print('file {} label is not paired!'.format(label_file))
        return None

    num = len(asr_text)
    doc_embed = []
    doc_spk = []
    doc_label = []
    doc_time = []
    doc_mel = []
    pre_label = 0
    for sent_idx in range(num):
        sentence = asr_text[sent_idx]
        logger.debug("info:")
        logger.debug(sentence)

        if len(sentence)==0:
            continue
        time, text = re.split(": spk., |: speak., ", sentence)
        start_time, end_time = transform_time_string(time)

        doc_time.append((start_time, end_time))
        # get sentence embedding
        sent_embedding = get_sentence_embedding(text, sent_idx, model)
        logger.debug('sent embedding dim {}'.format(len(sent_embedding)))
        doc_embed.append(sent_embedding)
        sent_spk = get_sentence_speaker(sentence)
        logger.debug('spk code: {}'.format(sent_spk))
        doc_spk.append(sent_spk)
        sent_mel = capture_interval(waveform, start_time, end_time, sample_rate)
        doc_mel.append(sent_mel)
        sent_label = get_sentence_label(text, labels, pre_label)
        pre_label = sent_label
        logger.debug('sent label: {}'.format(sent_label))
        doc_label.append(sent_label)
        if sent_idx % 100 == 0:
            logger.info('process {} of {}'.format(sent_idx, num))
    datasets = {
        "name": name,
        "text_embedding": doc_embed,
        "time": doc_time,
        "spk": doc_spk,
        "label": doc_label,
        "mel": doc_mel
    }
    return datasets


def run():
    args = arg_parse()
    logger.info(args)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    data_dir = os.path.join(args.data_dir, args.dataset)
    file_name_list = os.listdir(os.path.join(data_dir, 'asr'))
    num_file = len(file_name_list)

    # load models
    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1', device=args.device)
    model.eval()

    datasets = []
    save_ind = 0
    for idx, name in tqdm(enumerate(file_name_list)):
        logger.info('process file {}'.format(name))
        asr_file = os.path.join(data_dir, 'asr', name)
        label_file = os.path.join(data_dir, 'label', name)
        audio_file = os.path.join(data_dir, 'audio', name.replace(".json", ".wav"))
        if not os.path.exists(label_file):
            continue
        doc_datasets = get_docs_embedding(name, asr_file, label_file, audio_file, model)
        if doc_datasets is None:
            logger.warning('doc: name {} is None'.format(name))
            continue
        datasets.append(doc_datasets)
        if idx % 20 == 0:
            logger.info('******* process {} of {} files ********'.format(idx, num_file))
        if (idx + 1) % args.batch_size == 0:
            save_name = os.path.join(args.save_dir, args.dataset + '.' + str(save_ind) + '.pt')
            save_ind += 1
            torch.save(datasets, save_name)
            datasets = []
    if len(datasets) > 0:
        save_name = os.path.join(args.save_dir, args.dataset + '.' + str(save_ind) + '.pt')
        save_ind += 1
        torch.save(datasets, save_name)


if __name__ == '__main__':
    run()
