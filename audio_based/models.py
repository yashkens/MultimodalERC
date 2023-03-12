import torch
import wandb
from sklearn.metrics import f1_score
from torch.nn import DataParallel


class EmotionClassificationModel:

    def __init__(self, model, num_classes, device, parallel=False):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.parallel = parallel

        if self.parallel:
            self.model = DataParallel(self.model).to(device)
        else:
            self.model.to(device)

    def __call__(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            output = self.model(inputs)
            logits = output['logits']
            pred = torch.argmax(logits, dim=1)
        return pred

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss, val_fscore = 0, 0

        with torch.no_grad():
            for batch in val_dataloader:

                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.model(inputs, labels=labels)
                loss = output['loss']
                logits = output['logits']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                val_loss += loss.item()

                pred = torch.argmax(logits, dim=1)

                fscore = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                val_fscore += fscore

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_f1 = val_fscore / len(val_dataloader)
        return avg_val_loss, avg_val_f1

    def train(self, train_dataloader, val_dataloader, n_epoch, optimizer, model_save_name, save=True, patience=3):

        train_losses, val_losses = [], []
        train_fscores, val_fscores = [], []
        val_scores = []
        no_improv_epochs = 0

        for epoch in range(n_epoch):

            self.model.train()

            train_loss, train_fscore = 0, 0
            for step_num, batch in enumerate(train_dataloader):

                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                output = self.model(inputs, labels=labels)
                loss = output['loss']
                logits = output['logits']

                if self.parallel:
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean()

                train_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

                pred = torch.argmax(logits, dim=1)

                fscore = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='weighted')
                train_fscore += fscore

                wandb.log({"batch train loss": loss.item()})

            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_f1 = train_fscore / len(train_dataloader)

            avg_val_loss, avg_val_f1 = self.validate(val_dataloader)

            train_losses.append(avg_train_loss)
            train_fscores.append(avg_train_f1)
            val_losses.append(avg_val_loss)
            val_fscores.append(avg_val_f1)

            if max(val_fscores) > avg_val_f1:
                no_improv_epochs += 1
            else:
                if save:
                    if self.parallel:
                        torch.save(self.model.module.state_dict(), model_save_name)
                    else:
                        torch.save(self.model.state_dict(), model_save_name)

            if no_improv_epochs > patience:
                return None

            wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss,
                       "train F1": avg_train_f1, "val F1": avg_val_f1,
                       "epoch": epoch})
        return None
