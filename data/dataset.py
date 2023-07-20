import numpy as np
from torch.utils.data import Dataset

class eegDataset(Dataset):
    def __init__(self, data, label, transform=None):
        super().__init__()
        self.data = data
        self.labels = label
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is None:
            data = self.data[index]
        else:
            data = self.transform(self.data[index])
        
        label = self.labels[index]
        
        return data, label
    
    def __len__(self):
        return len(self.data)

    def combineDataset(self, otherDataset):
        self.labels = np.hstack((self.labels, otherDataset.labels))
        self.data = np.concatenate((self.data, otherDataset.data), axis=0)