"""
Created on Aug 27 2021

@author: Guan-Ting Lin
"""

def compare(pred_start, pred_end, gold_start, gold_end):
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
    
    return overlap_start, overlap_end, Min, Max


def Frame_F1_score(pred_starts, pred_ends, gold_starts, gold_ends):
    F1s = []
    for pred_start, pred_end, gold_start, gold_end in zip(pred_starts, pred_ends, gold_starts, gold_ends):
        overlap_start, overlap_end, Min, Max = compare(pred_start, pred_end, gold_start, gold_end)
        
        Precision = (overlap_end - overlap_start) / (pred_end - pred_start)
        Recall = (overlap_end - overlap_start) / (gold_end - gold_start)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        F1s.append(F1)
    
    return sum(F1s) / len(F1s)

def AOS_score(pred_starts, pred_ends, gold_starts, gold_ends):
    AOSs = []
    for pred_start, pred_end, gold_start, gold_end in zip(pred_starts, pred_ends, gold_starts, gold_ends):
        overlap_start, overlap_end, Min, Max = compare(pred_start, pred_end, gold_start, gold_end)
    
        AOS = (overlap_end - overlap_start) / (Max - Min)
        AOSs.append(AOS)

    return sum(AOSs) / len(AOSs)