import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np 

# --------------------- DATASET ---------------------

class EnergyDatasetFromRows(Dataset):
    def __init__(self, save_path, input_len=1790, output_len=240, train_split=0.8, train=True):
        self.df = pd.read_csv(save_path, header=0, sep=',', decimal=",")
        self.data = self.df.values.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len

        # Split the data into train and test sets
        split_idx = int(len(self.data) * train_split)
        if train:
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        x = row[:self.input_len]
        y = row[self.input_len:self.input_len + self.output_len]
        return torch.tensor(x), torch.tensor(y)

