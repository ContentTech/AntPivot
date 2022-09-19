# encoding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import argparse
import re
import jieba
import logging
import torch

from sentence_transformers import SentenceTransformer


logger = logging.getLogger()
logger.setLevel(logging.INFO)

INGORE_WORD = set(['，', '的'])
spk_map = {"speak0": 0,
               "spk1": 1,
               "spk2": 2
               }

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", default='data/')
    parser.add_argument('-dataset', default='debug', help='train, valid or test ')
    parser.add_argument('-batch_size', default=100,type=int, help='save batch size')
    parser.add_argument("-save_dir", default='data/output')
    parser.add_argument("-device", default='cpu')

    parser.add_argument("-logs_file", default='../logs/preprocess.log')
    args = parser.parse_args()
    return args


def get_sentence_speaker(sentence):
    sub_sentence = sentence[sentence.find('s'):]
    spk_info = sub_sentence[:sub_sentence.find(',')]
    logger.debug('speaker info: {}'.format(spk_info))
    return spk_map.get(spk_info, 0)


def get_sentence_label(text, labels):
    for label in labels:
        if text in label:
            label_type = label[:label.find(";")]
            assert label_type in ['start', 'end'], 'label {} not in (start, end)'.format(label_type)
            if label_type == 'start':
                return 1
            else:
                return 2
    return 0


def get_sentence_embedding(text, sent_idx, model):
    new_text = ''.join([word for word in jieba.lcut(text) if word not in INGORE_WORD])
    embeddings = model.encode([new_text], batch_size=1, show_progress_bar=False)
    # logger.debug('embeding data:')
    # logger.debug(embeddings)
    return embeddings[0]


def get_docs_embedding(asr_file, label_file, model=None):
    asr_text = json.load(open(asr_file))
    labels = json.load(open(label_file))

    num = len(asr_text)
    doc_embed = []
    doc_spk = []
    doc_label = []
    for sent_idx in range(num):
        sentence = asr_text[sent_idx]
        logger.debug("info:")
        logger.debug(sentence)

        if len(sentence)==0:
            continue
        time, text = re.split(": spk., |: speak., ", sentence)
        # get sentence embedding
        sent_embedding = get_sentence_embedding(text, sent_idx, model)
        logger.debug('sent embedding dim {}'.format(len(sent_embedding)))
        doc_embed.append(sent_embedding)
        sent_spk = get_sentence_speaker(sentence)
        logger.debug('spk code: {}'.format(sent_spk))
        doc_spk.append(sent_spk)
        sent_label = get_sentence_label(text, labels)
        logger.debug('sent label: {}'.format(sent_label))
        doc_label.append(sent_label)
        if sent_idx % 100 == 0:
            logger.info('process {} of {}'.format(sent_idx, num))
    datasets = {
        "text_embedding": doc_embed,
        "spk": doc_spk,
        "label": doc_label
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
    for idx, name in enumerate(file_name_list):
        logger.info('process file {}'.format(name))
        asr_file = os.path.join(data_dir, 'asr', name)
        label_file = os.path.join(data_dir, 'label', name)
        if not os.path.exists(label_file):
            continue
        doc_datasets = get_docs_embedding(asr_file, label_file, model)
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
