import numpy as np
import pandas as pd

import cv2
import scipy.io.wavfile
import os

import torch
from torch.utils.data import Dataset, DataLoader


class MultimodalEmotionDataset(Dataset):
    def __init__(self, data, label2id, clip_len=16, frame_sample_rate=4):
        self.label_dict = label2id
        self.data = data
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['Video_Path']
        audio_path = self.data.iloc[idx]['Audio_Path']
        
        encoding = self.get_frames(video_path)
        encoding = torch.Tensor(np.transpose(encoding, (0, 3, 1, 2)))
        
        sample_rate, audio = scipy.io.wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 2**15
        selected_frames_audio = self.sample_frame_indices(audio.shape[0],
                                                         frame_sample_rate=sample_rate//25,
                                                         mode="audio")
        audio = torch.Tensor(audio[None, selected_frames_audio])
        audio = audio.unsqueeze(2)
        
        label = self.label_dict[self.data.iloc[idx]['emotion']]
        label_vector = torch.zeros([6])
        label_vector[label] = 1
        
        return {'image': encoding, 'audio': audio, 'label': label_vector}

    def __len__(self):
        return len(self.data)
    
    def sample_frame_indices(self, seg_len, clip_len=16, frame_sample_rate=4, mode="video"):
        # seg_len -- how many frames are received
        # clip_len -- how many frames to return
        converted_len = int(clip_len * frame_sample_rate)
        converted_len = min(converted_len, seg_len-1)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        if mode == "video":
            indices = np.linspace(start_idx, end_idx, num=clip_len)
        else:
            indices = np.linspace(start_idx, end_idx, num=clip_len*frame_sample_rate)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
    
    def get_frames(self, file_path):
        cap = cv2.VideoCapture(file_path)
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self.sample_frame_indices(v_len)

        frames = []
        for fn in range(v_len):
            success, frame = cap.read()
            if success is False:
                continue
            if (fn in indices):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                res = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                frames.append(res)
        cap.release()
            
        if len(frames) < self.clip_len:
            add_num = self.clip_len - len(frames)
            frames_to_add = [frames[-1]] * add_num
            frames.extend(frames_to_add)

        return np.array(frames)
    
    
def fullfill_path(df: pd.DataFrame, video_path: str, audio_path: str) -> pd.DataFrame:
    df['Video_Path'] = video_path + df.file_path
    df.file_path = df.file_path.apply(lambda x: x[:-4])
    df['Audio_Path'] = audio_path + df.file_path + '.wav'
    return df
    
def prepare_data(bs: int, csv_path: str='/cephfs/home/mikhaylova/multimodal_emo_reco/CREMA-D/CSV',
                video_path: str='/cephfs/home/mikhaylova/multimodal_emo_reco/CREMA-D/VideoFlash/',
                audio_path: str='/cephfs/home/mikhaylova/AudioWAV-CREMA/'):
    head_folder = csv_path
    train = pd.read_csv(f'{head_folder}/train.csv')
    val = pd.read_csv(f'{head_folder}/val.csv')
    test = pd.read_csv(f'{head_folder}/test.csv')
    
    train = fullfill_path(train, video_path, audio_path)
    val = fullfill_path(val, video_path, audio_path)
    test = fullfill_path(test, video_path, audio_path)
    
    labels = list(set(train['emotion']))
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    
    train_dataset = MultimodalEmotionDataset(train, label2id)
    test_dataset = MultimodalEmotionDataset(test, label2id)
    val_dataset = MultimodalEmotionDataset(val, label2id)
    
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    
    return len(labels), label2id, id2label, train_dataloader, test_dataloader, val_dataloader