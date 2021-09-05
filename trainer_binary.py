"""
Created on Aug 26 2021

@author: Guan-Ting Lin
"""
import torch

from torch.utils.data import DataLoader
# from model import TransformerQA
from model_binary import TransformerQA
from data import SpokenSquadTrainDataset, train_collate_fn
from utils import Frame_F1_score, AOS_score
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pad_sequence
import os
import wandb

class trainer():
    def __init__(self, config, paras):
        self.config = config
        self.paras = paras
        self.n_epoch = self.config['hparas']['n_epoch']
        self.exp_name = paras.name
        self.log_interval = self.paras.log_interval
        self.valid_interval = self.paras.valid_interval
        self.batch_size = self.config['hparas']['batch_size']

        os.makedirs(paras.ckptdir, exist_ok=True)
        self.ckptdir = os.path.join(paras.ckptdir,self.exp_name)
        os.makedirs(self.ckptdir, exist_ok=True)

        os.makedirs(paras.logdir, exist_ok=True)
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        os.makedirs(self.logdir, exist_ok=True)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.extracter = torch.hub.load('s3prl/s3prl', self.paras.upstream, 
            feature_selection=self.config['data']['feature_selection']).to(self.device)
        self.extracter.to(self.device)
        self.extracter.eval()
        self.max_position_embeddings = self.config['model']['max_position_embeddings']
        self.downsample_factor = self.config['data']['downsample_factor']

    def to_device(self, feat):
        return [f.to(self.device) for f in feat]

    def load_data(self):
        
        self.train_dataset = SpokenSquadTrainDataset(
            self.paras.upstream,
            self.paras.n_worker,
            self.device, 
            self.config['split'][0],
            **self.config['data']
        )
        # self.dev_dataset = SpokenSquadTrainDataset(
        #     self.paras.upstream,
        #     self.paras.n_worker,
        #     self.device, 
        #     self.config['split'][1],
        #     **self.config['data']
        # )

        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=train_collate_fn, 
            num_workers=self.paras.n_worker
        )
        # self.dev_loader = DataLoader(
        #     self.dev_dataset, 
        #     batch_size=1,
        #     shuffle=False, 
        #     collate_fn=collate_fn,
        #     num_workers=self.paras.n_worker
        # )
    
    def prepare_data(self, data):
        question_wavs, context_wavs, start_positions, end_positions, is_possible = data
        question_wavs = self.to_device(question_wavs)
        context_wavs = self.to_device(context_wavs)    

        def preprocess_qa(question_feature, context_feature, start_positions, end_positions):
            q_len = question_feature.size(0)
            c_len = context_feature.size(0)
            join_len = c_len + q_len
            if join_len > self.max_position_embeddings:
                print(f'[INFO]   discard len { c_len + q_len}')
                join_len = self.max_position_embeddings
            
            segment_ids = torch.zeros(join_len, dtype=torch.long)
            segment_ids[q_len:] = 1
            position_ids = torch.arange(join_len, dtype=torch.long)
            qa_pair_feat = torch.cat((question_feature, context_feature[:join_len - q_len]), dim=0)
            start_positions = start_positions // self.downsample_factor
            end_positions = end_positions // self.downsample_factor
            if start_positions > 0 and end_positions > 0:
                start_positions += q_len
                end_positions += q_len
            
            return qa_pair_feat, segment_ids, position_ids, start_positions, end_positions

        qa_pair_feats, segment_ids, position_ids, src_key_padding_masks, new_start_positions, new_end_positions = [], [], [], [], [], []
        for i in range(len(question_wavs)):
            with torch.no_grad():
                question_feature = self.extracter([question_wavs[i]])
                context_feature = self.extracter([context_wavs[i]])
                question_feature = question_feature['default'][0]
                context_feature = context_feature['default'][0]

            # downsampling
            if self.downsample_factor is not None: 
                question_feature = question_feature[::self.downsample_factor]
                context_feature = context_feature[::self.downsample_factor]

            qa_pair_feat, segment_id, position_id, start_position, end_position = preprocess_qa(question_feature, context_feature, start_positions[i], end_positions[i])
            src_key_padding_mask = torch.ones(segment_id.size(), dtype=torch.bool)
            qa_pair_feats.append(qa_pair_feat)
            segment_ids.append(segment_id)
            position_ids.append(position_id)
            src_key_padding_masks.append(src_key_padding_mask)
            new_start_positions.append(start_position)
            new_end_positions.append(end_position)

            
        # padding batch
        qa_pair_feats = pad_sequence(qa_pair_feats, batch_first=True, padding_value=0) 
        src_key_padding_masks = ~pad_sequence(src_key_padding_masks, batch_first=True, padding_value=0)
        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0) 
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0) 
        is_possible = torch.tensor(list(is_possible))
        cls_index = torch.zeros(is_possible.size(), dtype=torch.long)
        return (
            qa_pair_feats.to(self.device), 
            src_key_padding_masks.to(self.device), 
            segment_ids.to(self.device), 
            position_ids.to(self.device), 
            is_possible.to(self.device),
            cls_index.to(self.device),
        )


    def set_model(self):
        self.model = TransformerQA(**self.config['model'])
        self.model.to(self.device)

        self.optim = eval('torch.optim.' + self.config['hparas']['optimizer'])(
                        self.model.parameters(), 
                        lr=float(self.config['hparas']['lr'])
                    )
    
    def exec(self):
        self.step = 0
        # self.best_F1_score = 0.0
        # self.best_AOS_score = 0.0

        wandb.watch(self.model)
        for epoch in tqdm(range(self.n_epoch), desc='epoch'):
            # training
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                
                self.optim.zero_grad()
                feat, src_key_padding_mask, segment_ids, position_ids, is_possible, cls_index = self.prepare_data(batch)

                total_loss, cls_logits = self.model(
                    feat_embs=feat, 
                    position_ids=position_ids,
                    segment_ids=segment_ids, 
                    src_key_padding_mask=src_key_padding_mask,
                    cls_index=cls_index,
                    is_possible=is_possible
                )

                total_loss.backward()
                self.optim.step()
                pred = torch.sigmoid(cls_logits)
                for i in range(len(pred)):
                    if pred[i] > 0.5: 
                        pred[i] = 1
                    else: 
                        pred[i] = 0
                correct = pred == is_possible
                acc = torch.sum(correct) / len(correct)
                # TODO: logger
                wandb.log({'train_cls_loss': total_loss})
                wandb.log({'train_acc': acc})
                print(f'[INFO]  train_cls_loss: {total_loss}, acc:{acc}')
                del feat, src_key_padding_mask, segment_ids, position_ids
                # del total_loss, cls_loss, start_loss, end_loss, start_logits, end_logits
                self.step  = self.step + 1

            # validation
            # self.validate()
                
        # print(f'[INFO]   Best F1 score: {self.best_F1_score}')
        # print(f'[INFO]   Best AOS score: {self.best_AOS_score}')



    def validate(self): 
        self.model.eval()
        dev_Frame_F1_scores, dev_AOS_scores = [], []
        print(f'[INFO]   validation at step {self.step}')
        for i, data in tqdm(enumerate(self.dev_loader)):
            feat, src_key_padding_mask, segment_ids, position_ids, is_possible, cls_index = self.prepare_data(data)
            
            with torch.no_grad():
                total_loss, cls_logits = self.model(
                    feat_embs=feat, 
                    position_ids=position_ids,
                    segment_ids=segment_ids, 
                    src_key_padding_mask=src_key_padding_mask,
                    cls_index=cls_index,
                    is_possible=is_possible
                )

                pred = torch.sigmoid(cls_logits)
                for i in range(len(pred)):
                    if pred[i] > 0.5: 
                        pred[i] = 1
                    else: 
                        pred[i] = 0
                correct = pred == is_possible
                acc = torch.sum(correct) / len(correct)
                # TODO: logger
                wandb.log({'dev_cls_loss': total_loss})
                wandb.log({'dev_acc': acc})
        
        
        # save ckpt if get better score
        if acc > self.best_F1_score: 
            ckpt_path = os.path.join(self.ckptdir, 'best_F1_score.pt')
            full_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optim.state_dict(),
                'epoch': epoch,
                'f1_score': dev_Frame_F1_score,
                'AOS_score':dev_AOS_score
            }
            torch.save(full_dict, ckpt_path)
            self.best_F1_score = dev_Frame_F1_score
        
        if dev_AOS_score > self.best_AOS_score: 
            ckpt_path = os.path.join(self.ckptdir, 'best_AOS_score.pt')
            full_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optim.state_dict(),
                'epoch': epoch,
                'f1_score': dev_Frame_F1_score,
                'AOS_score':dev_AOS_score
            }
            torch.save(full_dict, ckpt_path)
            self.best_AOS_score = dev_AOS_score




