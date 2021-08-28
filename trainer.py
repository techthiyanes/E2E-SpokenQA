"""
Created on Aug 26 2021

@author: Guan-Ting Lin
"""
import torch

from torch.utils.data import DataLoader
from transformers import AdamW
from model import TransformerQA
from data import SpokenSquadDataset, collate_fn
from utils import Frame_F1_score, AOS_score
from tqdm import tqdm
from torch.nn.functional import softmax
import os

class trainer():
    def __init__(self, config, paras):
        self.config = config
        self.paras = paras
        self.n_epoch = self.config['hparas']['n_epoch']
        self.exp_name = paras.name

        os.makedirs(paras.ckptdir, exist_ok=True)
        self.ckptdir = os.path.join(paras.ckptdir,self.exp_name)
        os.makedirs(self.ckptdir, exist_ok=True)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def fetch_data(self, batch, device):
        for data in batch: 
            data = data.to(device)
        return batch

    def load_data(self):
        
        self.train_dataset = SpokenSquadDataset(
            self.paras.upstream,
            self.device, 
            **self.config['data']
        )
        self.dev_dataset = SpokenSquadDataset(
            self.paras.upstream,
            self.device, 
            **self.config['data']
        )

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['hparas']['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        self.dev_loader = DataLoader(
            self.dev_dataset, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=collate_fn
        )

    def set_model(self):
        self.model = TransformerQA(**self.config['model'])
        self.model.to(self.device)
        print(self.model)

        self.optim = eval('torch.optim.' + self.config['hparas']['optimizer'])(
                        self.model.parameters(), 
                        lr=float(self.config['hparas']['lr'])
                    )
    def exec(self):
        best_F1_score = 0.0
        best_AOS_score = 0.0

        for epoch in tqdm(range(self.n_epoch), desc='epoch'):
            # training
            self.model.train()
            for batch in self.train_loader:
                self.optim.zero_grad()
                feat, src_key_padding_mask, segment_ids, position_ids, start_positions, end_positions = self.fetch_data(batch, self.device)

                loss, start_logits, end_logits = self.model(
                    feat_embs=feat, 
                    position_ids=position_ids,
                    segment_ids=segment_ids, 
                    src_key_padding_mask=src_key_padding_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss.backward()
                self.optim.step()
                
                # TODO: logger
                pred_start_prob = softmax(start_logits, dim=1)
                pred_end_prob = softmax(end_logits, dim=1)
                pred_start_positions = torch.argmax(pred_start_prob, dim=1)
                pred_end_positions = torch.argmax(pred_end_prob, dim=1)

                F1 = Frame_F1_score(pred_start_positions, pred_end_positions, start_positions, end_positions)
                AOS = AOS_score(pred_start_positions, pred_end_positions, start_positions, end_positions)
            
            # inference
            self.model.eval()
            dev_Frame_F1_scores, dev_AOS_scores = [], []

            for i, data in enumerate(self.dev_loader):
                feat, src_key_padding_mask, segment_ids, position_ids, start_positions, end_positions = self.fetch_data(data, self.device)
                
                with torch.no_grad():
                    loss, start_logits, end_logits = self.model(
                    feat_embs=feat, 
                    position_ids=position_ids,
                    segment_ids=segment_ids, 
                    src_key_padding_mask=src_key_padding_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                    )
                
                
                # TODO: logger
                pred_start_prob = softmax(start_logits, dim=1)
                pred_end_prob = softmax(end_logits, dim=1)
                pred_start_positions = torch.argmax(pred_start_prob, dim=1)
                pred_end_positions = torch.argmax(pred_end_prob, dim=1)

                F1 = Frame_F1_score(pred_start_positions, pred_end_positions, start_positions, end_positions)
                AOS = AOS_score(pred_start_positions, pred_end_positions, start_positions, end_positions)

                dev_Frame_F1_scores.append(F1)
                dev_AOS_scores.append(AOS)
            
            dev_Frame_F1_score = sum(dev_Frame_F1_scores) / len(dev_Frame_F1_scores)
            dev_AOS_score = sum(dev_AOS_scores) / len(dev_AOS_scores)
            
            # save ckpt if get better score
            if dev_Frame_F1_score > best_F1_score: 
                ckpt_path = os.path.join(self.ckptdir, 'best_F1_score.pt')
                full_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                    'epoch': epoch,
                    'f1_score': dev_Frame_F1_score,
                    'AOS_score':dev_AOS_score
                }
                torch.save(full_dict, ckpt_path)
                best_F1_score = dev_Frame_F1_score
            
            if dev_AOS_score > best_AOS_score: 
                ckpt_path = os.path.join(self.ckptdir, 'best_AOS_score.pt')
                full_dict = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                    'epoch': epoch,
                    'f1_score': dev_Frame_F1_score,
                    'AOS_score':dev_AOS_score
                }
                torch.save(full_dict, ckpt_path)
                best_AOS_score = dev_AOS_score

        print(f'[INFO]   Best F1 score: {best_F1_score}')
        print(f'[INFO]   Best AOS score: {best_AOS_score}')


