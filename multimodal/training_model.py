import wandb
import torch
from sklearn.metrics import f1_score


def get_preds(data: torch.Tensor, true: torch.Tensor,
              model: torch.nn.Module):
    """Evaluate model"""
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        __, predicted = torch.max(outputs, 1)
    return predicted

def validate(model: torch.nn.Module,
             dev: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module):
    """Validate model"""
    model.eval()
    val_loss, val_fscore = 0, 0

    with torch.no_grad():
        for batch in dev:
            inputs, labels = batch
            
            outputs = model(inputs)
            labels = labels.to(dtype=torch.int64)
            loss = loss_fn(outputs, labels)
           
            val_loss += loss.item()

            pred = get_preds(inputs, labels, model)
            gold = labels.view(-1)
            fscore = f1_score(gold.cpu().numpy(), pred.cpu().numpy(), 
                              average='weighted')
            val_fscore += fscore

    avg_val_loss = val_loss / len(dev)
    avg_val_f1 = val_fscore / len(dev)
    return avg_val_loss, avg_val_f1

def train(epochs: int,
          model: torch.nn.Module,
          train: torch.utils.data.DataLoader,
          dev: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          patience: int):
    """Train model"""
    prev_val_score = 0
    no_improv_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_fscore = 0, 0
        for data in train:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.to(dtype=torch.int64)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pred = get_preds(inputs, labels, model)
            gold = labels.view(-1)
            
            fscore = f1_score(gold.cpu().numpy(), pred.cpu().numpy(), 
                              average='weighted')
            train_fscore += fscore

            wandb.log({"batch train loss": loss.item()})

        avg_train_loss = train_loss / len(train)
        avg_train_f1 = train_fscore / len(train)

        # EARLY STOPPING CODE
        avg_val_loss, avg_val_f1 = validate(model, dev, loss_fn)
        if avg_val_f1 < prev_val_score:
            no_improv_epochs += 1
        prev_val_score = avg_val_f1

        if no_improv_epochs >= patience:
            return None

        wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss,
            "train F1": avg_train_f1, "val F1": avg_val_f1,
            "epoch": epoch})

    return None
