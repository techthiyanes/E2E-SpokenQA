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

class TransformerQA(nn.Module):

    def __init__(self, hidden_size: int=768, num_attention_heads: int=4,
                 nlayers: int=6, hidden_dropout_prob: float=0.1, intermediate_size: int=2048, 
                 hidden_act: str='gelu', layer_norm_eps=1e-12, num_labels: int=2):
        super().__init__()
        layer_norm_eps = float(layer_norm_eps)
        
        self.embedding = ExtraEncoding(
                max_position_embeddings=2000, 
                segment_type_size=2, 
                hidden_size=hidden_size, 
                hidden_dropout_prob=hidden_dropout_prob, 
                layer_norm_eps=layer_norm_eps)

        encoder_layers = TransformerEncoderLayer(
                    hidden_size, num_attention_heads, intermediate_size, hidden_dropout_prob,
                    hidden_act, layer_norm_eps, batch_first=True)

        # TransformerEncoder is a stack of N encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.qa_predictor = nn.Linear(hidden_size, num_labels)

        # TODO: init_weight

    def forward(self, feat_embs, position_ids, segment_ids, src_key_padding_mask, start_positions, end_positions):
        """
        Args:

        Returns:
            
        """
        output_emb = self.embedding(feat_embs, position_ids, segment_ids)
        encoder_output = self.transformer_encoder(output_emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.qa_predictor(encoder_output)

        # (B, T, 2) -> (B, T, 1) -> (B, T)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits

# add segment emb + positional emb
class ExtraEncoding(nn.Module):
    def __init__(self, 
                max_position_embeddings=2000, 
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

