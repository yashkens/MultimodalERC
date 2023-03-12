import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, load_metric
from custom_datasets import CREMADataset
from transformers import Wav2Vec2FeatureExtractor


def test(test_dataloader, answer_model):
    gold, pred = [], []
    for batch in tqdm(test_dataloader):
        inputs, labels = batch
        res = answer_model(inputs)
        pred.extend(res.tolist())
        gold.extend(labels.tolist())
    fscore = f1_score(gold, pred, average='weighted')
    return fscore


def prepare_data(bs, model_id):
    data_files = {
        "train": '/cephfs/home/dolidze/anaconda3/ERC/CREMA Audio/train_audio.csv',
        "test": '/cephfs/home/dolidze/anaconda3/ERC/CREMA Audio/test_audio.csv',
        "validation": '/cephfs/home/dolidze/anaconda3/ERC/CREMA Audio/val_audio.csv',
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\,", )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    eval_dataset = dataset["validation"]

    label_list = train_dataset.unique("emotion")
    label_list.sort()
    num_labels = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    train = CREMADataset(train_dataset, label2id, feature_extractor)
    dev = CREMADataset(eval_dataset, label2id, feature_extractor)
    test = CREMADataset(test_dataset, label2id, feature_extractor)

    train_dataloader = DataLoader(train, batch_size=bs, shuffle=True)
    dev_dataloader = DataLoader(dev, batch_size=bs, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=bs, shuffle=False)
    return num_labels, label2id, id2label, train_dataloader, dev_dataloader, test_dataloader


