import wandb

import torch
import random
import numpy as np
from transformers import HubertForSequenceClassification, AutoConfig
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils import test, prepare_data, prepare_data_meld, total_freeze
from models import EmotionClassificationModel


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--name", default="hubert-audio-classification", type=str, help="project name for wandb")
parser.add_argument("--lr", default=1e-4, help="learning rate")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--seed", default=42, type=int, help="random seed value")
parser.add_argument("--data", default='CREMA', type=str)
parser.add_argument("--encoder", default='facebook/hubert-base-ls960', type=str)
parser.add_argument("--trainable_layers", default='all', type=str)
# parser.add_argument("--parallel", default='parallel', type=str)

def train_net(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        encoder_name = config.encoder.replace('facebook/', '')
        name_str = f"{config.data}-{encoder_name}_seed-{config.seed}-" \
                   f"trainable_{config.trainable_layers}-{config.parallel}--{config.lr}"
        run.name = name_str
        model_id = config.encoder

        if config.data == 'MELD':
            num_labels, label2id, id2label, train_dataloader, dev_dataloader, test_dataloader =\
                prepare_data_meld(config.batch_size, model_id)
        else:
            num_labels, label2id, id2label, train_dataloader, dev_dataloader, test_dataloader = \
                prepare_data(config.batch_size, model_id)
        config_m = AutoConfig.from_pretrained(model_id, num_labels=num_labels)
        hubert_model = HubertForSequenceClassification.from_pretrained(
            model_id,
            config=config_m,
            ignore_mismatched_sizes=True
        )

        parallel = False
        if config.parallel == 'parallel':
            parallel = True

        optimizer = torch.optim.Adam(params=hubert_model.parameters(), lr=config.lr)
        save_name = name_str + '.pt'
        
        if config.trainable_layers == 'none':
            hubert_model = total_freeze(hubert_model)

        answer_model = EmotionClassificationModel(hubert_model, num_classes=num_labels, device='cuda',
                                                  parallel=parallel)

        patience = 3
        epochs = config.epochs
        answer_model.train(train_dataloader, dev_dataloader, epochs, optimizer, save_name, save=False,
                           patience=patience)

        test_fscore = test(test_dataloader, answer_model)
        wandb.log({"test F1": test_fscore})


def create_config(args):
    sweep_config = {'method': 'grid'}
    metric = {
        'name': 'val F1',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'parallel': {
            'values': ['non-parallel', 'parallel']
        }
    }
    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'epochs': {
            'value': 20
        },
        'lr': {
            'value': args.lr
        },
        'batch_size': {
            'value': args.batch_size
        },
        'encoder': {
            'value': args.encoder
        },
        'data': {
            'value': args.data
        },
        'seed': {
            'value': args.seed
        },
        'trainable_layers': {
            'value': args.trainable_layers
        }
    })

    return sweep_config


if __name__ == "__main__":
    wandb.login()

    args = parser.parse_args()
    sweep_config = create_config(args)
    sweep_id = wandb.sweep(sweep_config, project=args.name)
    wandb.agent(sweep_id, train_net)
    sweep_config = {'method': 'grid'}

    metric = {
        'name': 'val F1',
        'goal': 'maximize'
        }

    sweep_config['metric'] = metric
