import torch
import random
import re
import numpy as np
import wandb

from models import resnet
from train import train, test_step
from datasets import prepare_data


def load_pretrained_model(layers: int, 
                          num_classes: int,
                          device: str, 
                          model_name: str) -> torch.nn.Module:
    model = resnet.generate_model(layers, n_classes=num_classes)
    checkpoint = torch.load(model_name + '.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model

def freeze_without_last_sequential(model: torch.nn.Module) -> torch.nn.Module:
    for name, param in model.named_parameters():
        if name.startswith("layer4"): 
            continue
        else:
            param.requires_grad = False
    return model

def train_net(config=None):
    
    with wandb.init(config=config) as run:

        config = wandb.config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        name_str = f"{config.model}_{config.seed}-{config.lr}"
        run.name = name_str
        
        num_labels, label2id, id2label, train_dataloader, test_dataloader, dev_dataloader, test_dataset = prepare_data(
            config.batch_size
        )
        
        device = 'cuda'
        
        layers = int(re.match(r'r3d(\d+)', config.model).group(1))
        dict_num_classes = {'K': 700, 'KM': 1039, 'KMS': 1139, 'KS': 800, 'M': 339, 'S': 100, 'MS': 439}
        num_classes = dict_num_classes[re.match(r'r3d\d+_(\w+)_', config.model).group(1)]
        
        model = load_pretrained_model(layers, num_classes, device, config.model)
        
        # Define loss and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
        
        # Freeze all except last one
        model = freeze_without_last_sequential(model).to(device)
        
        # Get the length of class_names (one output unit for each class)
        output_shape = num_labels

        # Recreate the classifier layer and seed it to the target device
        model.fc = torch.nn.Linear(in_features=2048, out_features=output_shape, bias=True).to(device)
        
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
