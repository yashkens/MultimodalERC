import wandb
import torch
from transformers import AutoProcessor, XCLIPVisionModel, AutoModel
from sklearn.metrics import f1_score
from torch.nn import DataParallel
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import logging
logging.set_verbosity_error()


# TODO: fix and import modality-specific models instead of copy-pasting
class TextClassificationModel:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def __call__(self, input_ids, attn_mask, return_last_hidden_state=False):

        self.model.eval()

        with torch.no_grad():

            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attn_mask,
                                output_hidden_states=return_last_hidden_state)
            logits = output['logits']
            pred = torch.argmax(logits, dim=1)
            if return_last_hidden_state:
                hidden_states = output['hidden_states']
        if return_last_hidden_state:
            return pred, hidden_states[-1][:, 0, :]
        else:
            return pred


class XCLIPClassificaionModel(nn.Module):
    def __init__(self, num_labels):
        super(XCLIPClassificaionModel, self).__init__()

        self.base_model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
        self.num_labels = num_labels

        hidden_size = self.base_model.config.hidden_size
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss()

        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, pixel_values, labels=None, return_last_hidden_state=False):

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        out = self.base_model(pixel_values)[0]  # [48, 50, 768]
        out = torch.transpose(out, 1, 2)  # [48, 768, 50]
        out = self.pool1(out)  # [48, 768, 1]
        out = torch.transpose(out, 1, 2)  # [48, 1, 768]
        out = out.squeeze(1)  # [48, 768]
        hidden_out = out.view(batch_size, num_frames, -1)  # [3, 16, 768]
        hidden_out = torch.transpose(hidden_out, 1, 2)  # [3, 768, 16]
        pooled_out = self.pool2(hidden_out)  # [3, 768, 1]
        pooled_out = torch.transpose(pooled_out, 1, 2)  # [3, 1, 768]

        pooled_out = pooled_out[:, 0, :]  # [3, 768]

        logits = self.classifier(pooled_out)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if return_last_hidden_state:
            return {'logits': logits, 'loss': loss, 'last_hidden_state': pooled_out}
        else:
            return {'logits': logits, 'loss': loss}


class VideoClassificationModel:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def __call__(self, pixel_values, return_last_hidden_state=False):

        self.model.eval()

        with torch.no_grad():

            pixel_values = pixel_values.to(self.device)

            output = self.model(pixel_values, return_last_hidden_state=return_last_hidden_state)
            logits = output['logits']
            pred = torch.argmax(logits, dim=1)
            if return_last_hidden_state:
                hidden_states = output['last_hidden_state']
        if return_last_hidden_state:
            return pred, hidden_states
        else:
            return pred


class ConvNet(nn.Module):
    def __init__(self, num_labels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(3186 * 32, 128)
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x, return_last_hidden_state=False):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flat(x)
        hid = torch.relu(self.fc1(x))
        x = self.fc2(hid)
        if not return_last_hidden_state:
            return {'logits': x}
        else:
            return {'logits': x, 'last_hidden_state': hid}


class AudioClassificationModel:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def __call__(self, input_ids, return_last_hidden_state=False):

        self.model.eval()

        with torch.no_grad():

            input_ids = torch.tensor(input_ids, dtype=torch.float).to(self.device)
            output = self.model(input_ids, return_last_hidden_state=return_last_hidden_state)
            logits = output['logits']
            pred = torch.argmax(logits, dim=1)
            if return_last_hidden_state:
                hidden_state = output['last_hidden_state']
        if return_last_hidden_state:
            return pred, hidden_state
        else:
            return pred


class MultimodalClassificaionModel(nn.Module):
    def __init__(self, text_model, video_model, audio_model, num_labels, input_size, hidden_size=256):
        super(MultimodalClassificaionModel, self).__init__()

        self.text_model = text_model
        self.video_model = video_model
        self.audio_model = audio_model
        self.num_labels = num_labels

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, self.num_labels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        # self.linear3 = nn.Linear(hidden_size // 2, self.num_labels)
        # self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()
        # self.drop1 = nn.Dropout()
        # self.drop2 = nn.Dropout()
        self.loss_func = CrossEntropyLoss()

    def forward(self, batch, labels=None):
        text_pred, text_last_hidden = self.text_model(
            batch['text']['input_ids'].squeeze(1),
            batch['text']['attention_mask'].squeeze(1),
            return_last_hidden_state=True
        )
        video_pred, video_last_hidden = self.video_model(
            batch['video']['pixel_values'].squeeze(1),
            return_last_hidden_state=True
        )
        audio_pred, audio_last_hidden = self.audio_model(
            batch['audio'],
            return_last_hidden_state=True
        )
        concat_input = torch.cat((text_last_hidden, video_last_hidden, audio_last_hidden), dim=1)

        hidden_state = self.linear1(concat_input)
        hidden_state = self.drop1(self.relu1(hidden_state))
        logits = self.linear2(hidden_state)
        # hidden_state = self.linear1(concat_input)
        # hidden_state = self.drop1(self.relu1(hidden_state))
        # hidden_state = self.linear2(hidden_state)
        # hidden_state = self.drop2(self.relu2(hidden_state))
        # logits = self.linear3(hidden_state)
        # logits = self.linear1(concat_input)

        loss = None
        if labels is not None:
            loss = self.loss_func(logits.view(-1, self.num_labels), labels.view(-1))

        return {'logits': logits, 'loss': loss}


class MainModel:

    def __init__(self, model, device, parallel=False):
        self.model = model
        self.device = device
        self.parallel = parallel
        self.model.to(device)

        if self.parallel:
            self.model = DataParallel(self.model).to(device)
        else:
            self.model.to(device)

    def __call__(self, batch):
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)
            logits = output['logits']
            pred = torch.argmax(logits, dim=1)
        return pred

    def validate(self, val_dataloader):

        self.model.eval()

        val_loss, val_fscore = 0, 0

        with torch.no_grad():
            for batch in val_dataloader:

                labels = batch['label']
                labels = labels.to(self.device)
                output = self.model(batch, labels=labels)
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

    def train(self, train_dataloader, val_dataloader, n_epoch, optimizer, model_save_name, save=False, patience=3):

        train_losses, val_losses = [], []
        train_fscores, val_fscores = [], []
        val_scores = []
        no_improv_epochs = 0

        for epoch in range(n_epoch):

            self.model.train()

            train_loss, train_fscore = 0, 0
            for step_num, batch in enumerate(train_dataloader):

                labels = batch['label']
                labels = labels.to(self.device)
                output = self.model(batch, labels=labels)
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
