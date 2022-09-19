import os
import glob
import random

import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


START = 1
END = 3


class BatchFortuneData(Dataset):
    def __init__(self, data_dir, datasets_name, **kwargs):
        self.names = glob.glob(os.path.join(data_dir, datasets_name + '.*.pt'))
        print('load {} number of files'.format(len(self.names)))
        self.data = []
        self.collect_data()

    def __len__(self):
        return len(self.data)

    def collect_data(self):
        for name in tqdm(self.names):
            try:
                local_data = torch.load(name)
                for item in local_data:
                    # pos_seg, neg_seg = segment_sampling(item['label'])
                    self.data.append({
                        "text_embedding": np.array(item['text_embedding'], dtype=float),
                        "spk": np.array(item['spk'], dtype=int),
                        "label": np.array(item['label'], dtype=int),
                        "time": np.array(item['time'], dtype=float),
                        # "time": np.array(item['time'], dtype=float),
                        "mel": np.array(item['mel'], dtype=float)
                        # "pos_seg": np.array(pos_seg, dtype=int),
                        # "neg_seg": np.array(neg_seg, dtype=int)
                    })
            except Exception as e:
                print(name)
                raise e

    def __getitem__(self, index):
        return self.data[index]

def segment_sampling(labels):
    pos_segs, neg_segs = [], []
    label_num = len(labels)
    start_idx, end_idx = -1, -1
    for outer_idx, label in enumerate(labels):
        start_idx = outer_idx if label == START else start_idx
        end_idx = outer_idx if label == END and start_idx != -1 else end_idx
        if start_idx != -1 and end_idx != -1:
            for inner_idx in range(start_idx, end_idx):
                if inner_idx - 1 >= start_idx:
                    pos_segs.append([inner_idx, random.randint(start_idx, inner_idx - 1)])
                if inner_idx + 1 <= end_idx:
                    pos_segs.append([inner_idx, random.randint(inner_idx + 1, end_idx)])
                if start_idx > 0:
                    neg_segs.append([inner_idx, random.randint(0, start_idx - 1)])
                if end_idx + 1 < label_num - 1:
                    neg_segs.append([inner_idx, random.randint(end_idx + 1, label_num - 1)])
            start_idx, end_idx = -1, -1
    assert (len(pos_segs) != 0 and len(neg_segs) != 0), "Invalid input!"
    return pos_segs, neg_segs

def calc_mel_difference(spk, mel):
    # L & [L, W, D_mel]
    mask_0, mask_1, mask_2 = (spk == 0), (spk == 1), (spk == 2)
    if mask_0.sum() != 0:
        mel[mask_0] = mel[mask_0] - mel[mask_0].mean(0)
    if mask_1.sum() != 0:
        mel[mask_1] = mel[mask_1] - mel[mask_1].mean(0)
    if mask_2.sum() != 0:
        mel[mask_2] = mel[mask_2] - mel[mask_2].mean(0)
    return mel

def collate_data(max_text_num, text_embd_dims, max_mel_width, mel_embd_dim, **kwargs):
    def collate_data_fn(samples):
        bsz = len(samples)

        timestamp = np.zeros((bsz, max_text_num, 2)).astype(np.float32)
        text_embedding = np.zeros((bsz, max_text_num, text_embd_dims)).astype(np.float32)
        mask = np.zeros((bsz, max_text_num)).astype(np.int64)
        spk = np.zeros((bsz, max_text_num)).astype(np.int64)
        label = np.zeros((bsz, max_text_num)).astype(np.int64)
        mel = np.zeros((bsz, max_text_num, max_mel_width, mel_embd_dim)).astype(np.float32)
        # pos_seg = np.zeros((bsz, max_sample_num, 2)).astype(np.int64)
        # neg_seg = np.zeros((bsz, max_sample_num, 2)).astype(np.int64)


        for i, sample in enumerate(samples):
            assert (len(sample["text_embedding"]) == len(sample["spk"]) and
                    len(sample["spk"]) == len(sample["label"])), "Invalid Inputs!"
            text_num = min(len(sample["text_embedding"]), max_text_num)
            timestamp[i, :text_num] = sample["time"][:text_num]
            text_embedding[i, :text_num] = sample["text_embedding"][:text_num]
            mask[i, :text_num] = 1

            # mel_diff = calc_mel_difference(sample["spk"], sample["mel"])
            spk[i, :text_num] = sample["spk"][:text_num]
            label[i, :text_num] = sample["label"][:text_num]
            mel[i, :text_num] = sample["mel"][:text_num]
            # mel[i, :text_num] = mel_diff[:text_num]


            # pos_len, neg_len = len(sample["pos_seg"]), len(sample["neg_seg"])
            # pos_idx = np.random.choice(pos_len, max_sample_num)
            # neg_idx = np.random.choice(neg_len, max_sample_num)
            # pos_seg[i], neg_seg[i] = sample["pos_seg"][pos_idx], sample["neg_seg"][neg_idx]
            # print("pos_seg:", pos_seg[i])
            # print("neg_seg:", neg_seg[i])

        return {
            "inputs": {
                "sent_emb": torch.from_numpy(text_embedding),
                "spk": torch.from_numpy(spk),
                "mask": torch.from_numpy(mask),
                "mel": torch.from_numpy(mel)
            },
            "target": {
                "tags": torch.from_numpy(label),
                # "pos_seg": torch.from_numpy(pos_seg),
                # "neg_seg": torch.from_numpy(neg_seg)
            },
            "raw": torch.from_numpy(timestamp)
        }

    return collate_data_fn
