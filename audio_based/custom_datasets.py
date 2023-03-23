from torch.utils.data import Dataset
import librosa

class MELDDataset(Dataset):
    def __init__(self, data, label_dict, feature_extractor):
        self.feature_extractor = feature_extractor
        self.label_dict = label_dict
        self.data = data

    def __getitem__(self, idx):
        file_name = self.data['Filename'][idx]
        audio_file = f"/cephfs/home/dolidze/anaconda3/ERC/Audio_MELD/Audio/train/{file_name}"
        audio_array = librosa.load(audio_file, sr=16000, mono=True)[0]

        encoding = self.feature_extractor(
            librosa.to_mono(audio_array),
            sampling_rate=16000,
            padding='max_length',
            max_length=50000,
            return_tensors="pt",
            truncation=True
        )

        encoding = encoding['input_values'].squeeze(0)
        label = self.label_dict[self.data['Emotion'][idx]]
        return encoding, label

    def __len__(self):
        return len(self.data)

class CREMADataset(Dataset):
    def __init__(self, data, label_dict, feature_extractor):
        self.feature_extractor = feature_extractor
        self.label_dict = label_dict
        self.data = data

    def __getitem__(self, idx):
        file_name = self.data['file_path'][idx]
        audio_file = f"/cephfs/home/dolidze/anaconda3/ERC/CREMA Audio/AudioWAV-CREMA/{file_name}"
        audio_array = librosa.load(audio_file, sr=16000, mono=True)[0]

        encoding = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            padding='max_length',
            max_length=50000,
            return_tensors="pt",
            truncation=True
        )

        encoding = encoding['input_values'].squeeze(0)
        label = self.label_dict[self.data['emotion'][idx]]
        return encoding, label

    def __len__(self):
        return len(self.data)
