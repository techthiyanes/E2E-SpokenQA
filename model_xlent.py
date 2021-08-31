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
from utils import PoolerStartLogits, PoolerEndLogits

class TransformerQA(nn.Module):

    def __init__(self, hidden_size: int=768, num_attention_heads: int=4,
                 nlayers: int=6, hidden_dropout_prob: float=0.1, intermediate_size: int=2048, 
                 hidden_act: str='gelu', layer_norm_eps=1e-12, num_labels: int=2, 
                 max_position_embeddings: int=8000, segment_type_size: int=2):
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
        self.start_logits = PoolerStartLogits(hidden_size)
        self.end_logits = PoolerEndLogits(hidden_size)

        # TODO: init_weight

    def forward(self, feat_embs, position_ids, segment_ids, src_key_padding_mask, start_positions, end_positions):
        """
        Args:

        Returns:
            
        """
        output_emb = self.embedding(feat_embs, position_ids, segment_ids)
        hidden_states = self.transformer_encoder(output_emb, src_key_padding_mask=src_key_padding_mask)
        
        start_logits = self.start_logits(hidden_states)
        
        if start_positions is not None and end_positions is not None:
            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            return total_loss, start_logits, end_logits

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = nn.functional.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index

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

