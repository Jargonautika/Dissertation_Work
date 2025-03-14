import torch
from torch.utils.data import Dataset

class customDataLoader(Dataset):
  
  def __init__(self, data, labels, last_idxes = None):
    super().__init__()
    self.data = data
    self.labels = labels
    self.oneHotLabels = torch.nn.functional.one_hot(self.labels.long()).float() # Turns it into vectors of [1, 0] or [0, 1]
    self.idxes = last_idxes
  

  def __getitem__(self, index):
    data = torch.tensor(self.data[index], dtype = torch.float)
    labels = torch.tensor(self.labels[index]) # Here we want a boolean value
    oneHotLabels = torch.tensor(self.oneHotLabels[index])
    if self.idxes != None:
        idxes = torch.tensor(self.idxes[index])
        return {"data": data, "labels": labels, "oneHotLabels": oneHotLabels, "indices": idxes}
    else: 
        return {"data": data, "labels": labels, "oneHotLabels": oneHotLabels}
  
  
  def __len__(self):
    return len(self.data)