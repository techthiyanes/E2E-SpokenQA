import torch
import os
import argparse
import yaml
# from trainer_binary import trainer
# from trainer import trainer
from trainer_passage import trainer
import wandb

parser = argparse.ArgumentParser(description='Training Spoken QA.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--ckptdir', default='ckpt/', type=str, help='Checkpoint path.', required=False)
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--log_interval', default=30, type=int, help='Log per batch.', required=False)
parser.add_argument('--valid_interval', default=2000, type=int, help='validation per batch.', required=False)
parser.add_argument('--n_worker', default=0, type=int, help='Number of workers for multprocessing', required=False)
parser.add_argument(
    '--upstream',
    help='Specify the upstream variant according to torch.hub.list'
)

paras = parser.parse_args()
config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

# wandb.init(project='SpokenQA', entity='daniel091444', name=paras.name, config=config)
# wandb.init(project='binary_QA', entity='daniel091444', name=paras.name, config=config)
wandb.init(project='passage_QA', entity='daniel091444', name=paras.name, config=config)

trainer = trainer(config, paras)
trainer.load_data()
trainer.set_model()
trainer.exec()

