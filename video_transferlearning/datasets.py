import numpy as np
import pandas as pd

import cv2
import os

import torch
from torch.utils.data import Dataset, DataLoader


class ImageEmotionDataset(Dataset):
    def __init__(self, data, label2id, clip_len=16, frame_sample_rate=4):
        self.label_dict = label2id
        self.data = data        

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['file_path']
        file_path = f'/cephfs/home/yashkens/MultimodalERC/Video/MELDSpeakers/{file_path}'
        
        all_frames = os.listdir(file_path)
        selected_inds = self.sample_frame_indices(len(all_frames))
        selected_frames = [str(max(i, 0) + 1) + '.png' for i in selected_inds]
        
        video = []
        for frame in selected_frames:
            image_path = file_path + '/' + frame
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = cv2.resize(img, dsize=(230, 230), interpolation=cv2.INTER_CUBIC)
            video.append(res)
        
        encoding = np.array(video)
        encoding = torch.Tensor(np.transpose(encoding, (3, 0, 1, 2)))
        label = self.label_dict[self.data.iloc[idx]['Emotion']]
        
        return encoding, label

    def __len__(self):
        return len(self.data)
    
    def sample_frame_indices(self, seg_len, clip_len=16, frame_sample_rate=4):
        # seg_len -- how many frames are received
        # clip_len -- how many frames to return
        converted_len = int(clip_len * frame_sample_rate)
        converted_len = min(converted_len, seg_len-1)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
    
    
def prepare_data(bs):
    full_df = pd.read_csv('/cephfs/home/yashkens/MultimodalERC/Video/MELDSpeakers/MELDSpeakers.csv')

    train = full_df[full_df['part'] == 'train']
    val = full_df[full_df['part'] == 'dev']
    test = full_df[full_df['part'] == 'test']
    
    labels = list(set(full_df['Emotion']))
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    train_dataset = ImageEmotionDataset(train, label2id)
    test_dataset = ImageEmotionDataset(test, label2id)
    val_dataset = ImageEmotionDataset(val, label2id)
    
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    
    return len(labels), label2id, id2label, train_dataloader, test_dataloader, val_dataloader, test_dataset