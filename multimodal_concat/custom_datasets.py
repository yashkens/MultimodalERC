import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import logging
logging.set_verbosity_error()


class MultimodalDataset(Dataset):
    def __init__(self, text_data, video_data, tokenizer, video_feature_extractor, max_len, label_dict, video_dir,
                 data_part):
        self.text_data = text_data
        self.video_data = video_data
        self.tokenizer = tokenizer
        self.video_feature_extractor = video_feature_extractor
        self.max_len = max_len
        self.label_dict = label_dict
        self.video_dir = video_dir
        self.data_part = data_part

    def get_text_item(self, idx):
        text_cut = self.text_data.iloc[idx]
        input_tokens = text_cut['utterance']
        encoding = self.tokenizer(input_tokens,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')
        label = self.label_dict[text_cut['emotion']]
        return encoding, label, text_cut['dialog num'], text_cut['utt num']

    def get_video_item(self, idx, file_path):
        video_cut = self.video_data[self.video_data['file_path'] == file_path]
        if len(video_cut) == 0:
            return {"pixel_values": torch.zeros([1, 16, 3, 224, 224])}
        file_path = f'{self.video_dir}{file_path}'
        all_frames = os.listdir(file_path)
        selected_inds = self.sample_frame_indices(len(all_frames))
        selected_frames = [str(max(i, 0) + 1) + '.png' for i in selected_inds]
        video = []
        for frame in selected_frames:
            image_path = file_path + '/' + frame
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
        assert len(video) == 16, f'Somehow you got a wrong number of frames for video {file_path}!'
        encoding = self.video_feature_extractor(videos=video, return_tensors="pt")
        return encoding

    def __getitem__(self, idx):
        text_encoding, label, dialog_num, utt_num = self.get_text_item(idx)
        video_path = f"{self.data_part}/dia{dialog_num}_utt{utt_num}/cutFrames"
        video_encoding = self.get_video_item(idx, video_path)
        return {'text': text_encoding, 'video': video_encoding, 'label': label}

    def __len__(self):
        return len(self.text_data)

    def sample_frame_indices(self, seg_len, clip_len=16, frame_sample_rate=4):
        converted_len = int(clip_len * frame_sample_rate)
        converted_len = min(converted_len, seg_len - 1)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices
