import numpy as np
import pandas as pd

import cv2
import scipy.io.wavfile
import os

import torch
from torch.utils.data import Dataset, DataLoader


class ImageEmotionDataset(Dataset):
    def __init__(self, data, label2id, clip_len=16, frame_sample_rate=4):
        self.label_dict = label2id
        self.data = data        

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['Video_Path']
        audio_path = self.data.iloc[idx]['Audio_Path']
        
        all_frames = os.listdir(video_path)
        selected_inds = self.sample_frame_indices(len(all_frames))
        selected_frames = [str(max(i, 0) + 1) + '.png' for i in selected_inds]

        video = []
        for frame in selected_frames:
            image_path = video_path + '/' + frame
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            video.append(res)

        encoding = np.array(video)
        encoding = torch.Tensor(np.transpose(encoding, (0, 3, 1, 2)))
        
        sample_rate, audio = scipy.io.wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 2**15
        selected_frames_audio = self.sample_frame_indices(audio.shape[0],
                                                         frame_sample_rate=sample_rate//25,
                                                         mode="audio")
        audio = audio[None, selected_frames_audio, 0:1]
        
        label = self.label_dict[self.data.iloc[idx]['Emotion']]
        label_vector = torch.zeros([7])
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
    
    
def fullfill_path(df: pd.DataFrame, part: str) -> pd.DataFrame:
    df.Video_Path = '/cephfs/home/yashkens/MultimodalERC/Video/MELDSpeakers/' + df.Video_Path
    df.Audio_Path = f'/cephfs/home/chepel/Audio/MELD Audio/audio_{part}/' + df.Audio_Path
    return df.query('not Video_Path.isnull()')
    
def prepare_data(bs):
    train = pd.read_csv('/cephfs/home/mikhaylova/multimodal_emo_reco/MELD_csv/multimodal_train.csv', 
                        usecols=['Dialogue_ID', 'Utterance_ID', 
                                 'Utterance', 'Emotion', 'Video_Path',
                                 'Audio_Path'])
    val = pd.read_csv('/cephfs/home/mikhaylova/multimodal_emo_reco/MELD_csv/multimodal_dev.csv', 
                        usecols=['Dialogue_ID', 'Utterance_ID', 
                                 'Utterance', 'Emotion', 'Video_Path',
                                 'Audio_Path'])
    test = pd.read_csv('/cephfs/home/mikhaylova/multimodal_emo_reco/MELD_csv/multimodal_test.csv', 
                        usecols=['Dialogue_ID', 'Utterance_ID', 
                                 'Utterance', 'Emotion', 'Video_Path',
                                 'Audio_Path'])
    
    train = fullfill_path(train, 'train')
    val = fullfill_path(val, 'dev')
    test = fullfill_path(test, 'test')
    
    labels = list(set(train['Emotion']))
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
    
    return len(labels), label2id, id2label, train_dataloader, test_dataloader, val_dataloader