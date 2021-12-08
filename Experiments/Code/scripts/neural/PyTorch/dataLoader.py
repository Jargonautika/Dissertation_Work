import torch
from torch.utils.data import Dataset

class customDataLoader(Dataset):
  
  def __init__(self, data, labels, last_idxes = None):
    super().__init__()
    self.data = data
    self.labels = labels
    self.idxes = last_idxes
  

  def __getitem__(self, index):
    data = torch.tensor(self.data[index], dtype = torch.float)
    labels = torch.tensor(self.labels[index]) # Here we want a boolean value
    if self.idxes != None:
        idxes = torch.tensor(self.idxes[index])
        return {"data": data, "labels": labels, "indices": idxes}
    else: 
        return {"data": data, "labels": labels}
  
  
  def __len__(self):
    return len(self.data)