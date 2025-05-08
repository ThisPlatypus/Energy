import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GAN as mds
import os
import datetime 

class MyDataset(Dataset):
    """ Define the dataset """
    def __init__(self, numpy_array, range=(-0.98, 0.98)):
        if isinstance(numpy_array, pd.DataFrame):
            numpy_array = numpy_array.to_numpy()

        self.scaler = MinMaxScaler(feature_range=range)
        scaled_data = self.scaler.fit_transform(numpy_array[:, -72:])
        self.data = torch.from_numpy(scaled_data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def inverse_transform(self, scaled_data):
        # Convert back to original scale
        original_scale_data = self.scaler.inverse_transform(scaled_data)
        return original_scale_data

    
def dataloader(data_set, batch_size=2, shuffle=True):
    """Define the dataloader"""
    dataset = MyDataset(data_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset


def train_models(
        z_dim = 10,  
        model = '30m',
        lr = 0.0002,
        epochs = 300, 
        k = 5,
        batch_size = 64,
        path_data = r'/home/wxia/researchdrive/paper1/cleaned_data/uk_data_agg.csv',
        saving_path = r'/home/wxia/researchdrive/paper1/Models/models_agg/gan/nl_agg',
        save_control = True
        ):

    """
    This the the training function of the model
    z_dim: The dimension of noise
    model: model typles related to resoultion of the data we have 15 minutes (15m), 30m, and 60m
    lr: learning rate
    epochs: epochs
    k: training ratio (train discriminator once, then train generator 5 times)
    path_data: path of input data
    saving_path: path to save the data and model
    save_control: save the model or not
    """
    
    # load data & dataloader 
    data = pd.read_csv(path_data, header=0, sep=';', decimal=",", index_col="index").drop(columns=["Unnamed: 0"]) 
    dataloader_uk, scaler = dataloader(data, batch_size, shuffle=True)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the models
    gen = mds.Generator(model, z_dim).to(device)
    dis = mds.Discriminator(model).to(device)
    gen = nn.DataParallel(gen)
    dis = nn.DataParallel(dis)

    # define the optimizers
    optimizer_gen = torch.optim.Adam(gen.parameters(), lr=lr)
    optimizer_dis = torch.optim.Adam(dis.parameters(), lr=lr)

    # define the loss functions
    criterion = nn.BCELoss()
    
    print('Training start!!')
    # train the model
    for epoch in range(epochs):
        print(f'Epoch [{epoch+1}/{epochs}]')
        for i, real_data in enumerate(dataloader_uk):
            # train the discriminator
            z = torch.rand(real_data.shape[0], z_dim).to(device) # nosie
            fake_data = gen(z) # generate fake data
            
            pre_label_r = dis(real_data.to(device)) # predict the real data
            pre_label_f = dis(fake_data.detach()) # predict the fake data
            
            # calculate the loss
            loss_dis = criterion(pre_label_r, torch.ones_like(pre_label_r))
            loss_dis += criterion(pre_label_f, torch.zeros_like(pre_label_f))
            
            # update the discriminator
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
            
            # update the generator
            for ii in range(k):
                z = torch.rand(real_data.shape[0], z_dim).to(device) # nosie
                fake_data = gen(z) # generate fake data
                pre_label_f = dis(fake_data) # predict the fake data
                loss_gen = criterion(pre_label_f, torch.ones_like(pre_label_f))
                
                # update the generator
                optimizer_gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()

            # # store the losses
            p = real_data.detach().cpu().numpy()
            q = fake_data.detach().cpu().numpy()

                # Calculate distance between real and fake data
            distance = np.mean(np.abs(p - q))

            # Save the best model based on the smallest distance
            if epoch == 0 and i == 0:
                best_distance = distance
                best_gen_path = None

            if distance < best_distance:
                best_distance = distance
                best_gen_path = saving_path + f'/best_generator.pt'
                torch.save(gen.state_dict(), best_gen_path)
                path_gen = best_gen_path
                print(f"Best model saved with distance: {best_distance}")
            # save the models according to the distance
            if save_control:
                time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                path = saving_path+f'/models'
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(dis.state_dict(), path+f'/discriminator__{epoch}_{time}.pt')
                torch.save(gen.state_dict(), path+f'/generator__{epoch}_{time}.pt')
                
                        
    return scaler, path_gen