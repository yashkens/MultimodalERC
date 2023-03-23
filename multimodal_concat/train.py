import wandb

import torch
import random
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils import prepare_data, prepare_models, test
from models import MultimodalClassificaionModel, MainModel
from transformers import logging
logging.set_verbosity_error()


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--name", default="multimodal-concat", type=str, help="project name for wandb")
parser.add_argument("--lr", default=1e-5, help="learning rate")
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--seed", default=42, type=int, help="random seed value")


# TODO: add wandb configs as separate files
def create_config(args):
    sweep_config = {'method': 'grid'}

    metric = {
        'name': 'val F1',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'hidden_size': {
            'values': [256, 512, 1024]
        }
    }

    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'epochs': {
            'value': 10
        },
        'bs': {
            'value': args.batch_size
        },
        'seed': {
            'value': args.seed
        },
        'lr': {
            'value': args.lr
        }
    })

    return sweep_config


def train_net(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)

        name_str = f"concat-hidden_size_{config.hidden_size}"
        run.name = name_str

        num_labels, train_dataloader, test_dataloader, dev_dataloader = prepare_data(config.bs)
        text_model, video_model = prepare_models(num_labels)

        multi_model = MultimodalClassificaionModel(
            text_model,
            video_model,
            num_labels,
            input_size=1792,
            hidden_size=config.hidden_size
        )

        device = 'cuda'
        final_model = MainModel(multi_model, device=device)
        optimizer = torch.optim.Adam(params=multi_model.parameters(), lr=config.lr)

        final_model.train(
            train_dataloader,
            dev_dataloader,
            config.epochs,
            optimizer,
            'none.pt',
            save=False,
            patience=3)

        test_fscore = test(test_dataloader, final_model)
        wandb.log({"test F1": test_fscore})


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
