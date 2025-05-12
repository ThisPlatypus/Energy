import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np 

# --------------------- DATASET ---------------------

class EnergyDatasetFromRows(Dataset):
    def __init__(self, save_path, input_len=72, output_len=72, train_split=0.8, train=True):
        self.df = pd.read_csv(save_path, header=0, sep=',', decimal=",")
        self.data = self.df.values.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len

        # Split the data into train and test sets
        split_idx = int(len(self.data) * train_split)
        # Standardize the data
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.data = (self.data - self.mean) / self.std
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



class EnergyDataset2(Dataset):
    def __init__(self, save_path):
        # Load the DataFrame
        self.df = pd.read_csv(save_path, header=0, sep=';', decimal=",", index_col="index").drop(columns=["Unnamed: 0"])
        
        self.data = self.df.values.astype(np.float32)

        # Determine column indices for 'x' and 'y'
        self.x_indices = [i for i, col in enumerate(self.df.columns) if 'x' in col.lower()]
        self.y_indices = [i for i, col in enumerate(self.df.columns) if 'y' in col.lower()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use the precomputed indices to slice the NumPy array
        x = self.data[idx, self.x_indices]
        y = self.data[idx, self.y_indices]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

