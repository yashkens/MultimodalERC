import random
import re
import numpy as np
import wandb

from train import train, test_step
from datasets import prepare_data

import torch
import torch.nn as nn
from transformers import PerceiverForMultimodalAutoencoding

import accelerate

def change_model(model: torch.nn.Module,
                 label2id: dict,
                 id2label: dict) -> torch.nn.Module:
    # change label2id and id2label in config
    conf_change = {'label2id': label2id, 'id2label': id2label}
    new_config = model.perceiver.input_preprocessor.modalities.label.config.__dict__
    for k, v in conf_change.items():
        new_config[k] = v

    # change minimal padding so label vector becomes length of 704 to fit attention vector size
    model.perceiver.input_preprocessor.padding['label'] = nn.parameter.Parameter(torch.rand([1, 697]))
    # change minimal padding parameter 
    model.perceiver.input_preprocessor.min_padding_size = 303 

    # change output linear layer
    model.perceiver.output_postprocessor.modalities.label.classifier = nn.Linear(
        in_features=512, out_features=7, bias=True
    )
    return model

def train_net(config=None):
    
    with wandb.init(config=config) as run:

        config = wandb.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        name_str = f"mm_perceiver_{config.seed}-{config.lr}"
        run.name = name_str
        
        num_labels, label2id, id2label, train_dataloader, test_dataloader, dev_dataloader = prepare_data(
            config.batch_size
        )
        
        device = 'cuda'
        
        model = PerceiverForMultimodalAutoencoding.from_pretrained('deepmind/multimodal-perceiver', low_cpu_mem_usage=True)
        model = change_model(model, label2id, id2label)
        
        # Define loss and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        patience = 3
        train(model=model,
              train_dataloader=train_dataloader,
              val_dataloader=dev_dataloader,
              optimizer=optimizer,
              loss_fn=loss_fn,
              epochs=config.epochs,
              device=device,
              patience=patience)
        
        test_fscore = test_step(model=model,
                                dataloader=test_dataloader,
                                device=device)
        
        wandb.log({"test F1": test_fscore})
        
        save_name = 'saving_models/' + name_str + '.pt'
        model.load_state_dict(torch.load(save_name))
