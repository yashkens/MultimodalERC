from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
  def __init__(self, X_train, y_train):
    self.X = X_train
    self.y = y_train
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len
