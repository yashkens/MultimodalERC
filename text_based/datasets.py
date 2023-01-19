from torch.utils.data import Dataset


class EmotionLinesDataset(Dataset):
    def __init__(self, data, label_dict, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_dict = label_dict
        self.data = data

    def __getitem__(self, idx):
        input_tokens = self.data.iloc[idx]['utterance']
        encoding = self.tokenizer(input_tokens,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')
        label = self.label_dict[self.data.iloc[idx]['emotion']]
        return encoding, label

    def __len__(self):
        return len(self.data)
