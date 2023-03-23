import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification
from custom_datasets import MultimodalDataset
from models import TextClassificationModel, XCLIPClassificaionModel, VideoClassificationModel
from transformers import logging
logging.set_verbosity_error()


def add_utt_numeration(df):
    dialog_nums = df['dialog num'].unique().tolist()
    utt_ids = []
    for num in dialog_nums:
        sub_df = df[df['dialog num'] == num]
        for i in range(len(sub_df)):
            utt_ids.append(i)
    df['utt num'] = utt_ids
    return df


def prepare_data(bs):  # TODO: add paths to arguments
    # TEXT
    text_path = '/cephfs/home/yashkens/MultimodalERC/MeldCSV/'
    train_text = add_utt_numeration(pd.read_csv(text_path + 'train.csv'))
    test_text = add_utt_numeration(pd.read_csv(text_path + 'test.csv'))
    dev_text = add_utt_numeration(pd.read_csv(text_path + 'dev.csv'))

    text_model_name = 'bert-large-uncased'
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # VIDEO
    video_path = '/cephfs/home/yashkens/MultimodalERC/Video/MELDSpeakers/'
    full_video = pd.read_csv(video_path + 'MELDSpeakers.csv')

    train_video = full_video[full_video['part'] == 'train']
    test_video = full_video[full_video['part'] == 'test']
    dev_video = full_video[full_video['part'] == 'dev']

    video_model_name = "microsoft/xclip-base-patch32"
    video_feature_extractor = AutoProcessor.from_pretrained(video_model_name)

    # LABEL DICT
    num_labels = len(set(train_text['emotion']))
    labels = sorted(list(set(train_text['emotion'])))
    label_dict = {}
    for i in range(len(labels)):
        label_dict[labels[i]] = i

    # MULTIMODAL
    multi_train = MultimodalDataset(
        train_text,
        train_video,
        tokenizer,
        video_feature_extractor,
        max_len=128,
        label_dict=label_dict,
        video_dir=video_path,
        data_part='train'
    )
    multi_test = MultimodalDataset(
        test_text,
        test_video,
        tokenizer,
        video_feature_extractor,
        max_len=128,
        label_dict=label_dict,
        video_dir=video_path,
        data_part='test'
    )
    multi_dev = MultimodalDataset(
        dev_text,
        dev_video,
        tokenizer,
        video_feature_extractor,
        max_len=128,
        label_dict=label_dict,
        video_dir=video_path,
        data_part='dev'
    )

    # DATALOADERS
    train_dataloader = DataLoader(multi_train, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(multi_test, batch_size=bs, shuffle=True)
    dev_dataloader = DataLoader(multi_dev, batch_size=bs, shuffle=True)
    return num_labels, train_dataloader, test_dataloader, dev_dataloader


def prepare_models(num_labels, device='cuda'):  # TODO: add paths to arguments
    # TEXT
    text_model_name = 'bert-large-uncased'
    text_base_model = AutoModelForSequenceClassification.from_pretrained(
        text_model_name,
        num_labels=num_labels
    )

    save_name = '/cephfs/home/yashkens/MultimodalERC/Concatenation/checkpoints/bert-large-uncased_none_seed-42.pt'
    state_dict = torch.load(save_name)
    text_base_model.load_state_dict(state_dict)

    text_model = TextClassificationModel(text_base_model, device=device)

    # VIDEO
    video_base_model = XCLIPClassificaionModel(num_labels)

    save_name = '/cephfs/home/yashkens/MultimodalERC/Concatenation/checkpoints/XCLIP_Augmented.pt'
    state_dict = torch.load(save_name)
    video_base_model.load_state_dict(state_dict)

    video_model = VideoClassificationModel(video_base_model, device=device)
    return text_model, video_model


def test(test_dataloader, answer_model):
    gold, pred = [], []
    for batch in tqdm(test_dataloader):
        labels = batch['label']
        res = answer_model(batch)
        pred.extend(res.tolist())
        gold.extend(labels.tolist())
    fscore = f1_score(gold, pred, average='weighted')
    return fscore




