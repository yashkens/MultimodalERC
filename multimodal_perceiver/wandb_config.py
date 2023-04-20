import wandb

import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utility import train_net

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--name", default="va-perceiver", type=str, help="project name for wandb")
parser.add_argument("--lr", default=3e-3, help="learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")

def create_config(args):
    sweep_config = {'method': 'grid'}

    metric = {
        'name': 'val F1',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric
    
    parameters_dict = {
        'seed': {
            'values': [42, 13]
        }
    }

    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'epochs': {
            'value': 3
        },
        'batch_size': {
            'value': args.batch_size
        },
        'lr': {
            'value': args.lr
        }
    })

    return sweep_config

if __name__ == "__main__":
    wandb.login()

    args = parser.parse_args()
    sweep_config = create_config(args)
    sweep_id = wandb.sweep(sweep_config, project=args.name)
    wandb.agent(sweep_id, train_net)
