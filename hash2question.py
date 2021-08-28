"""
Created on Aug 24 2021

@author: Guan-Ting Lin
"""
import json
from pathlib import Path

"""
IO setting
"""
squad_path = 'train-v1.1.json'
output_path = 'train-hash2question.json'

squad_path = Path(squad_path)
with open(squad_path, 'rb') as f:
    squad_dict = json.load(f)

contexts = []
questions = []
answers = []
hash_ids = []
passage_inds = []
qa_inds = []
topic_inds = []

for topic_ind, topic in enumerate(squad_dict['data']):
    for passage_ind, passage in enumerate(topic['paragraphs']):
        context = passage['context']
        for qa_ind, qa in enumerate(passage['qas']):
            question = qa['question']
            hash_id = qa['id']
            hash_ids.append(hash_id)
            passage_inds.append(passage_ind)
            qa_inds.append(qa_ind)
            topic_inds.append(topic_ind)

print(f'[INFO]  length of hash_id: {len(hash_ids)}')

hash2question = {}
for i in range(len(qa_inds)):
    hash2question[hash_ids[i]] = f'question-{topic_inds[i]}_{passage_inds[i]}_{qa_inds[i]}'

with open(output_path, 'w') as fp:
    json.dump(hash2question, fp)

print('[INFO]   done')