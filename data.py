"""
Created on Aug 25 2021

@author: Guan-Ting Lin
"""
import torch 
from torch.utils.data import DataLoader, Dataset
from torch import nn
import pandas as pd
import json
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
# from p_tqdm import p_map
from multiprocessing import Pool

SAMPLE_RATE = 16000


def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()




class SpokenSquadDataset(Dataset): 
    def __init__(self, upstream: str, n_worker: int, device: str, mode: str, file_dir: str, file_path: str, hash2question_path: str, ext: str, feature_selection: int, downsample_factor: int):
        self.mode = mode
        if self.mode == 'train':
            file_dir = file_dir[0]
            file_path = file_path[0]
            hash2question_path = hash2question_path[0]
        else: 
            file_dir = file_dir[1]
            file_path = file_path[1]
            hash2question_path = hash2question_path[1]
        
        self.question_list, self.context_list, self.starts, self.ends = self.read_data(
            file_dir, file_path, hash2question_path, n_worker, ext
        )


    def read_file(self, file_path, hash2question_path):
        df = pd.read_csv(file_path)
        # remove duplicate answer 
        dup = df.duplicated(subset=['hash']) & df.duplicated(subset=['utterance'])
        dup = dup.values
        too_long = df['end'].values >= 25.0
        union = dup | too_long
        drop_idx = []
        for i in range(len(union)):
            if union[i]: 
                drop_idx.append(i)

        print(f'[INFO]  drop {len(drop_idx)} samples due to duplicated or too long')
        df = df.drop(drop_idx)

        with open(hash2question_path, 'r') as f:
            h2q = json.load(f)
        df['question'] = df['hash'].apply(lambda x: h2q[x])
        
        return(
            df['question'].values.tolist(),
            df['utterance'].values.tolist(),
            df['start'].values.tolist(),
            df['end'].values.tolist(),
        )

    def read_data(self, file_dir, file_path, hash2question_path, n_worker, ext='mp3', sr=16000):
        question_wavs, context_wavs = [], []
        question_list, context_list, starts, ends = self.read_file(file_path, hash2question_path)

        question_list = [file_dir + '/' + question_list[i] + '.' + ext for i in range(len(question_list))]
        context_list = [file_dir + '/' + context_list[i] + '.' + ext for i in range(len(context_list))]

        return(
            question_list,
            context_list,
            starts,
            ends,
        )

    def sec2ind(self, time):
        """
        for wav2vec 2.0 and Hubert: 
            sampling rate = 16000 Hz
            cnn encoder downsampling rate = 320
        
        1 sec = 16000 pt / 320 = 50 pt
        1 pt = 0.02 sec
        """
        return int(time / 0.02)


    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):

        return (
            self.question_list[idx],
            self.context_list[idx],
            self.sec2ind(self.starts[idx]),
            self.sec2ind(self.ends[idx]),
        )

def collate_fn(batch):
    question_list, context_list, start_idx, end_idx = zip(*batch)
    question_wavs, context_wavs = [], []
    # for i in range(len(question_list)):           
    #     question_wavs.append(reader(question_list[i]))
    #     context_wavs.append(reader(context_list[i]))
    question_wavs = [reader(question_file) for question_file in question_list]
    context_wavs = [reader(context_file) for context_file in context_list]
    return question_wavs, context_wavs, start_idx, end_idx

