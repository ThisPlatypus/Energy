import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import vae_models as mds
import os
import datetime 


class MyDataset(Dataset):
    """ Define the dataset """
    def __init__(self, numpy_array, range=(-0.98, 0.98)):
        if isinstance(numpy_array, pd.DataFrame):
            numpy_array = numpy_array.to_numpy()

        self.scaler_input = MinMaxScaler(feature_range=range)
        self.scaler_target = MinMaxScaler(feature_range=range)

        # Fit scalers on input and target data
        self.data = self.scaler_input.fit_transform(numpy_array[:, :1550])  # First 1550 columns as input
        self.target_data = self.scaler_target.fit_transform(numpy_array[:, -240:])  # Last 240 columns as target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32) 

    def inverse_transform(self, scaled_data, target=True):
        # Select the correct scaler
        scaler = self.scaler_target if target else self.scaler_input

        # Ensure the scaled_data shape matches the scaler's expectations
        if target and scaled_data.shape[1] != 240:
            raise ValueError(f"Expected scaled_data to have 240 features, but got {scaled_data.shape[1]} features.")
        if not target and scaled_data.shape[1] != 1550:
            raise ValueError(f"Expected scaled_data to have 1550 features, but got {scaled_data.shape[1]} features.")

        # Convert back to original scale
        original_scale_data = scaler.inverse_transform(scaled_data)
        return original_scale_data

    
def dataloader(data_set, batch_size=2, shuffle=True):
    """Define the dataloader"""
    dataset = MyDataset(data_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

def loss_function_vae(recon_x, x, mu, logvar):
    # Slice the target to match the reconstructed output
    x = x[:, -240:]  # Use only the last 240 columns as the target

    # Compute the reconstruction loss (MSE)
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # Compute the KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = mse + kld
    return loss, mse, kld


def train_models(
    input_dim = 48,
    hidden_dim = 10,
    path_data =  r'/home/wxia/researchdrive/paper1/cleaned_data/uk_data_agg.csv',
    saving_path =  r'/home/wxia/researchdrive/paper1/Models/models_agg/vae/uk_agg',
    lr = 0.0002,
    batch_size = 64,
    epochs = 400,
    save_control = False,
    model_typle = '30m',
    ):
        
    """
    This the the training function of the model
    z_dim: The dimension of noise
    model_typle: model typles related to resoultion of the data we have 15 minutes (15m), 30m, and 60m
    lr: learning rate
    epochs: epochs
    k: training ratio (train discriminator once, then train generator 5 times)
    path_data: path of input data
    saving_path: path to save the data and model
    save_control: save the model or not
    """
        
    # load the data & data loader
    
    data = pd.read_csv(path_data, header=0, sep=';', decimal=",", index_col="index").drop(columns=["Unnamed: 0"]).iloc[:, :72]

    data_loader, scaler = dataloader(data, batch_size=batch_size, shuffle=True)

    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the model
    if model_typle == '30m':
        vae = mds.VAE_30m(input_shape=input_dim, latent_dim=hidden_dim).to(device)
    elif model_typle == '60m':
        vae = mds.VAE_60m(input_shape=input_dim, latent_dim=hidden_dim).to(device)
    vae = nn.DataParallel(vae)

    # define the optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # store the loss
    losses = []
    msees = []
    klds = []

    # train the model
    print('Start training the model!')
    for epoch in range(epochs):
        print(f'Epoch [{epoch+1}/{epochs}]')
        for i, data in enumerate(data_loader):
            # renconstruct the data
            data = data.to(device).float()
            recon_data, mu, logvar = vae(data)
            
            # compute the loss
            loss, mse_l, kld_l = loss_function_vae(recon_data, data, mu, logvar) 
            
            #optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            msees.append(mse_l.item())
            klds.append(kld_l.item())
            
            # compute the js divergence
            p = data.cpu().detach().numpy()
            q = recon_data.cpu().detach().numpy()
            # Ensure p and q have the same shape
            p = p[:, -240:]  # Use only the last 240 columns of p to match q

            # Compute the distance
            distance = np.mean(np.abs(p - q))
            # Save the best model based on the smallest distance
            if epoch == 0 and i == 0:
                best_distance = distance
                best_gen_path = None

            if distance < best_distance:
                best_distance = distance
                best_gen_path = saving_path + f'/best_generator.pt'
                torch.save(vae.state_dict(), best_gen_path)
                path_gen = best_gen_path
                print(f"Best model saved with distance: {best_distance}")

            if save_control:
                print('saving the models')
                time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                path = saving_path+f'/models'
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(vae.module.state_dict(), path+f'/{model_typle}__{epoch}_{time}.pt')
                  
    return scaler, path_gen
    