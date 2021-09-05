"""
Created on Aug 27 2021

@author: Guan-Ting Lin
"""
import torch 
from torch import Tensor, device, nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.
    """

    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            :obj:`torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            x = x * (1 - p_mask) - 1e30 * p_mask

        return x



class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.
    """

    def __init__(self, hidden_size=768, layer_norm_eps=1e-12):
        super().__init__()
        self.dense_0 = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense_1 = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense_0 = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Position of the CLS token for each sentence in the batch. If :obj:`None`, takes the last token.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x

# metric calculation
def compare(pred_start, pred_end, gold_start, gold_end):
    if gold_start == 0.0 and gold_end == 0.0:
        no_answer = True
        overlap_start = None
        overlap_end = None
        Max = None
        Min = None
        no_overlap = None

    else: 
        no_answer = False
        if pred_start >= pred_end: 
            overlap_start = 0
            overlap_end = 0
            Max = 0
            Min = 0
            no_overlap = True
        elif pred_end <= gold_start or pred_start >= gold_end:
            overlap_start = 0
            overlap_end = 0
            Max = 0
            Min = 0
            no_overlap = True
        else:
            no_overlap = False
            if pred_start <= gold_start:
                Min = pred_start
                overlap_start = gold_start
            else: 
                Min = gold_start
                overlap_start = pred_start
            
            if pred_end <= gold_end:
                Max = gold_end
                overlap_end = pred_end
            else: 
                Max = pred_end
                overlap_end = gold_end
        
    return overlap_start, overlap_end, Min, Max, no_overlap, no_answer


def Frame_F1_score(pred_starts, pred_ends, gold_starts, gold_ends):
    F1s = []
    not_count = 0
    for pred_start, pred_end, gold_start, gold_end in zip(pred_starts, pred_ends, gold_starts, gold_ends):
        overlap_start, overlap_end, Min, Max, no_overlap, no_answer = compare(pred_start, pred_end, gold_start, gold_end)
        if no_answer: 
            not_count += 1

        else:
            if no_overlap: 
                F1 = 0
            else: 
                Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
                Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
                F1 = 2 * Precision * Recall / (Precision + Recall)
            F1s.append(F1)
    if len(F1s) - not_count > 0:
        return sum(F1s) / (len(pred_starts) - not_count)
    else:
        return 0

def AOS_score(pred_starts, pred_ends, gold_starts, gold_ends):
    AOSs = []
    not_count = 0
    for pred_start, pred_end, gold_start, gold_end in zip(pred_starts, pred_ends, gold_starts, gold_ends):
        overlap_start, overlap_end, Min, Max, no_overlap, no_answer = compare(pred_start, pred_end, gold_start, gold_end)
        if no_answer: 
            not_count += 1

        else:
            if no_overlap: 
                AOS = 0
            else: 
                AOS = (overlap_end - overlap_start) / (Max - Min)
            AOSs.append(AOS)
    if len(AOSs) - not_count > 0:    
        return sum(AOSs) / (len(pred_starts) - not_count)
    else: 
        return 0