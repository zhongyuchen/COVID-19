import torch
import pickle
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type('torch.FloatTensor')
        self.y = torch.from_numpy(y)
        assert len(self.x) == len(self.y)
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
