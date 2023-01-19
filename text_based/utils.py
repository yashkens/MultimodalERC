import pandas as pd
from torch.utils.data import DataLoader
from datasets import EmotionLinesDataset
from sklearn.metrics import f1_score


def prepare_data(tokenizer, data_path, bs, max_len):
    """
    Prepares datasets and data loaders from .csv files.
    Returns number of labels in data, three data loaders and test dataset (for testing).
    #TODO: probably can be improved to have a prettier return
    ----------
    tokenizer
        Tokenizer for the selected model.
    data_path : str
        Path to the directory with three data files (train, test and dev).
    bs: int
        Batch size.
    max_len: int
        Maximum sequence length. Used in tokenizer to truncate/pad sequences.
    """
    train_data = pd.read_csv(data_path + 'train.csv')
    test_data = pd.read_csv(data_path + 'test.csv')
    dev_data = pd.read_csv(data_path + 'dev.csv')

    num_labels = len(set(train_data['emotion']))
    labels = sorted(list(set(train_data['emotion'])))
    label_dict = {}
    for i in range(len(labels)):
        label_dict[labels[i]] = i

    train_dataset = EmotionLinesDataset(train_data, label_dict, tokenizer, max_len=max_len)
    dev_dataset = EmotionLinesDataset(dev_data, label_dict, tokenizer, max_len=max_len)
    test_dataset = EmotionLinesDataset(test_data, label_dict, tokenizer, max_len=max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    return num_labels, train_dataloader, test_dataloader, dev_dataloader, test_dataset


def test(test_dataset, answer_model):
    """
    Evaluates a model on a test set.
    ----------
    test dataset: EmotionLinesDataset
        Data as an EmotionLinesDataset object.
    answer_model
        Model to test. An EmotionClassificationModel object.
    """
    gold, pred = [], []
    for i in range(len(test_dataset)):
        res = answer_model(test_dataset[i][0]['input_ids'], test_dataset[i][0]['attention_mask'])
        pred.append(int(res))
        gold.append(test_dataset[i][1])
    fscore = f1_score(gold, pred, average='weighted')
    return fscore


def freeze_except_last_two(model):
    """
    Freezes model layers except for the last two and the classifier head.
    For now only works with BERT.
    TODO: avoid hard-coding layer names so it can be used with other encoders.
    """
    for name, param in model.named_parameters():
        if name.startswith("classifier") or name.startswith("bert.encoder.layer.11") or name.startswith("bert.encoder.layer.10"):
            continue
        else:
            param.requires_grad = False
    return model


def freeze_all(model):
    """
    Freezes all model layers except for the classifier head.
    For now only works with BERT.
    TODO: avoid hard-coding layer names so it can be used with other encoders.
    """
    for name, param in model.bert.named_parameters():
        param.requires_grad = False
    return model
