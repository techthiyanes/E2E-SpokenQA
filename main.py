import torch
import os
import argparse
import yaml
from trainer import trainer
import wandb

parser = argparse.ArgumentParser(description='Training Spoken QA.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--ckptdir', default='ckpt/', type=str, help='Checkpoint path.', required=False)
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--log_interval', default=10, type=int, help='Log per batch.', required=False)
parser.add_argument('--n_worker', default=8, type=int, help='Number of workers for multprocessing', required=False)
parser.add_argument(
    '--upstream',
    help='Specify the upstream variant according to torch.hub.list'
)

paras = parser.parse_args()
config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

wandb.init(project='SpokenQA', entity='daniel091444', name=paras.name)

trainer = trainer(config, paras)
trainer.load_data()
trainer.set_model()
trainer.exec()

