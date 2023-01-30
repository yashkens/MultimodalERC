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

    
class EmotionLinesDatasetWithContext(Dataset):
    def __init__(self, data, label_dict, tokenizer, max_len, n_context=3, sep='[SEP]', context_first=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_dict = label_dict
        self.data = data
        
        self.n_context = n_context
        if self.n_context > 10:
            self.n_context = 10
            
        self.sep = sep
        self.context_first = context_first

    def __getitem__(self, idx):
        dialog_num = self.data['dialog num'].iloc[idx]
        
        if idx > self.n_context:
            context_data = self.data.iloc[idx - self.n_context:idx]
        else:
            context_data = self.data.iloc[:idx]

        context = context_data[context_data['dialog num'] == dialog_num]['utterance']
        context = ' '.join(context)
        
        input_tokens = self.data.iloc[idx]['utterance']
        
        if self.context_first:
            input_context_query = self.sep.join([context, input_tokens])
        else:
            input_context_query = self.sep.join([input_tokens, context])
        
        encoding = self.tokenizer(input_context_query,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')
        label = self.label_dict[self.data.iloc[idx]['emotion']]
        
        return encoding, label

    def __len__(self):
        return len(self.data)