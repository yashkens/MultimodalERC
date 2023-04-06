import wandb

import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from utils import train_net

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--name", default="3d-cnn-resnet-meld", type=str, help="project name for wandb")
parser.add_argument("--lr", default=3e-3, help="learning rate")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--seed", default=42, type=int, help="random seed value")
parser.add_argument("--model", default='', type=str, help="model type")

def create_config(args):
    sweep_config = {'method': 'grid'}

    metric = {
        'name': 'val F1',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric
    
    if not args.model:
        parameters_dict = {
            'model': {
                'values': ['r3d50_K_200ep', 'r3d50_M_200ep', 'r3d50_S_200ep', 'r3d50_KS_200ep', 
                          'r3d50_MS_200ep', 'r3d50_KM_200ep', 'r3d50_KMS_200ep']
            }
        }
    else:
        parameters_dict = {
            'model': {
                'values': [args.model]
            }
        }

    sweep_config['parameters'] = parameters_dict

    parameters_dict.update({
        'epochs': {
            'value': 10
        },
        'batch_size': {
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

if __name__ == "__main__":
    wandb.login()

    args = parser.parse_args()
    sweep_config = create_config(args)
    sweep_id = wandb.sweep(sweep_config, project=args.name)
    wandb.agent(sweep_id, train_net)
