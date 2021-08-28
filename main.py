import torch
import os
import argparse
import yaml
from trainer import trainer

parser = argparse.ArgumentParser(description='Training Spoken QA.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--ckptdir', default='ckpt/', type=str, help='Checkpoint path.', required=False)
parser.add_argument(
    '--upstream',
    help='Specify the upstream variant according to torch.hub.list'
)

paras = parser.parse_args()
config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

trainer = trainer(config, paras)
trainer.load_data()
trainer.set_model()
trainer.exec()

