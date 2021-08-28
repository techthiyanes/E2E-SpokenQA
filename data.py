"""
Created on Aug 25 2021

@author: Guan-Ting Lin
"""
import torch 
import pytorch_lightning as pl
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

def read_file(file_path, hash2question_path):
    df = pd.read_csv(file_path)
    with open(hash2question_path, 'r') as f:
        h2q = json.load(f)
    df['question'] = df['hash'].apply(lambda x: h2q[x])
    
    return(
        df['question'].values.tolist(),
        df['utterance'].values.tolist(),
        df['start'].values.tolist(),
        df['end'].values.tolist(),
    )


def reader(fname):
    wav, ori_sr = torchaudio.load(fname)
    if ori_sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(ori_sr, SAMPLE_RATE)(wav)
    return wav.squeeze()




class SpokenSquadDataset(Dataset): 
    def __init__(self, upstream: str, n_worker: int, device: str, mode: str, file_dir: str, file_path: str, hash2question_path: str, ext: str, feature_selection: int):
        self.mode = mode
        if self.mode == 'train':
            file_dir = file_dir[0]
            file_path = file_path[0]
            hash2question_path = hash2question_path[0]
        else: 
            file_dir = file_dir[1]
            file_path = file_path[1]
            hash2question_path = hash2question_path[1]
        
        self.question_wavs, self.context_wavs, self.starts, self.ends = self.read_data(
            file_dir, file_path, hash2question_path, n_worker, ext
        )

        self.extracter = torch.hub.load('s3prl/s3prl', upstream, feature_selection=feature_selection).to(device)

    def read_data(self, file_dir, file_path, hash2question_path, n_worker, ext='mp3', sr=16000):
        question_wavs, context_wavs = [], []
        question_list, context_list, starts, ends = read_file(file_path, hash2question_path)

        question_list = [file_dir + '/' + question_list[i] + '.' + ext for i in range(len(question_list))]
        context_list = [file_dir + '/' + context_list[i] + '.' + ext for i in range(len(context_list))]
        p = Pool(n_worker)
        # question_wavs = p.map(reader, full_question_list)
        # context_wavs = p.map(reader, full_context_list)
        # print(question_wavs)
        # if self.close:
        # p.close()
        # p.join()
        for i in tqdm(range(len(question_list)), desc='reading data'):           
            question_wavs.append(reader(question_list[i]))
            context_wavs.append(reader(context_list[i]))

        return(
            question_wavs,
            context_wavs,
            starts,
            ends,
        )
    def get_feature(self, idx):
        # TODO: check whether exceeds SSL model max input length
        # TODO: extract which layer?
        with torch.no_grad():
            question_feature = self.extracter([self.question_wavs[idx]])
            context_feature = self.extracter([self.context_wavs[idx]])
        return question_feature['default'][0], context_feature['default'][0] 

    def sec2ind(self, time):
        """
        for wav2vec 2.0 and Hubert: 
            sampling rate = 16000 Hz
            cnn encoder downsampling rate = 320
        
        1 sec = 16000 pt / 320 = 50 pt
        1 pt = 0.02 sec
        """
        return int(time / 0.02)

    def preprocess_qa(self, question_feature, context_feature):
        q_len = question_feature.size(0)
        c_len = context_feature.size(0)
        segment_ids = torch.zeros(q_len + c_len, dtype=torch.long)
        segment_ids[q_len:] = 1
        position_ids = torch.arange(q_len + c_len, dtype=torch.long)
        qa_pair_feat = torch.cat((question_feature, context_feature), dim=0)
        
        return qa_pair_feat, segment_ids, position_ids


    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        question_feature, context_feature = self.get_feature(idx)
        qa_pair_feats, segment_ids, position_ids = self.preprocess_qa(question_feature, context_feature)

        return (
            qa_pair_feats,
            torch.ones(segment_ids.size(), dtype=torch.bool),
            segment_ids,
            position_ids, 
            self.sec2ind(self.starts[idx]),
            self.sec2ind(self.ends[idx]),
        )


# TODO: pad sequence into batch
def collate_fn(batch):
    """
    feat: (B, T, D)
    """
    feat, src_key_padding_mask, segment_ids, position_ids, start_positions, end_positions = zip(*batch)
    feat = pad_sequence(feat, batch_first=True, padding_value=0) 
    src_key_padding_mask = ~pad_sequence(src_key_padding_mask, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0) 
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0) 
    start_positions = torch.tensor(list(start_positions), dtype=torch.long)
    end_positions = torch.tensor(list(end_positions), dtype=torch.long)

    return feat, src_key_padding_mask, segment_ids, position_ids, start_positions, end_positions
