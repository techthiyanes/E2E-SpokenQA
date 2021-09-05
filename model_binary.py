"""
Created on Aug 25 2021

@author: Guan-Ting Lin
"""
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import CrossEntropyLoss
from utils import PoolerStartLogits, PoolerEndLogits, PoolerAnswerClass

class TransformerQA(nn.Module):

    def __init__(self, hidden_size: int=768, num_attention_heads: int=4,
                 nlayers: int=3, hidden_dropout_prob: float=0.1, intermediate_size: int=1024, 
                 hidden_act: str='gelu', layer_norm_eps=1e-12, num_labels: int=2, 
                 max_position_embeddings: int=1000, segment_type_size: int=2):
        super().__init__()
        layer_norm_eps = float(layer_norm_eps)
        
        self.embedding = ExtraEncoding(
                max_position_embeddings=max_position_embeddings, 
                segment_type_size=segment_type_size, 
                hidden_size=hidden_size, 
                hidden_dropout_prob=hidden_dropout_prob, 
                layer_norm_eps=layer_norm_eps)

        encoder_layers = TransformerEncoderLayer(
                    hidden_size, num_attention_heads, intermediate_size, hidden_dropout_prob,
                    hidden_act, layer_norm_eps, batch_first=True)

        # TransformerEncoder is a stack of N encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.answer_class = PoolerAnswerClass(hidden_size)
        self.dense_0 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(hidden_size, 1, bias=False)
    

        # TODO: init_weight

    def forward(self, feat_embs, position_ids, segment_ids, src_key_padding_mask, cls_index=None, is_possible=None):
        """
        Args:

        Returns:
            
        """
        output_emb = self.embedding(feat_embs, position_ids, segment_ids)
        hidden_states = self.transformer_encoder(output_emb, src_key_padding_mask=src_key_padding_mask)
      
   
        if cls_index is not None and is_possible is not None:
            # Predict answerability from the representation of CLS and START
            # cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
            output = self.dense_0(hidden_states[:, 0, :])
            output = self.activation(output)
            cls_logits = self.dense_1(output)
            loss_fct_cls = nn.BCEWithLogitsLoss()
            cls_logits = cls_logits.squeeze()
            cls_loss = loss_fct_cls(cls_logits, is_possible)


            return cls_loss, cls_logits

       

# add segment emb + positional emb
class ExtraEncoding(nn.Module):
    def __init__(self, 
                max_position_embeddings=10000, 
                segment_type_size=2, 
                hidden_size=768, 
                hidden_dropout_prob=0.1, 
                layer_norm_eps=1e-12):
        super().__init__()
        """
        from HuggingFace
        https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForQuestionAnswering
        """

        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_type_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "segment_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(self, feat_embs, position_ids, segment_ids):

        segment_embeddings = self.segment_embeddings(segment_ids)
        embeddings = feat_embs + segment_embeddings
        position_embeddings = self.position_embeddings(position_ids)
     
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

