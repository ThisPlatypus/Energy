import gan_tool as tg
import torch 
import torch.nn as nn
import GAN as mds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

TEST_PATH = "/home/chiara/Energy/Data/Test_2m_72.csv" 
CSV_PATH = "/home/chiara/Energy/Data/Train_2m_72.csv"  # ← Replace this with your actual file
MODEL_PATH = "/home/chiara/Energy/SAVED_MODEL/GAN"  # ← Replace this with your actual file
SAVE_CSV = "/home/chiara/Energy/PRED/gan_predictions"
while True:
    scaler, ganpaht = tg.train_models(
                z_dim = 72,
                model = '60m',
                lr = 0.0002,
                epochs = 2000, 
                k = 5,
                batch_size = 64,
                path_data = CSV_PATH,
                saving_path = MODEL_PATH,
                save_control = False
                )
    break
for i in range(50):
    # load the model
    #ganpaht = f'/home/chiara/Energy/SAVED_MODEL/GAN/models/generator__0_20250505-210340.pt' # The path of stored models
    gen = mds.Generator('60m', 72)
    state_dict = torch.load(ganpaht)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    gen.load_state_dict(new_state_dict)

    # original data

    or_data = np.array(pd.read_csv(TEST_PATH, header=0, sep=';', decimal=",", index_col="index").drop(columns=["Unnamed: 0"]))
    gen_data = gen(torch.rand(or_data.shape[0], 72)).detach().numpy()  # Input 1550 hours
    gen_data = scaler.inverse_transform(gen_data)



    # save the generated data
    # Save the generated data with the same index as the TEST data
    gen_df = pd.DataFrame(gen_data, index=pd.read_csv(TEST_PATH, header=0, sep=';', decimal=",", index_col="index").drop(columns=["Unnamed: 0"]).index)
    gen_df.to_csv(f'{SAVE_CSV}_{i}.csv', index_label="index")
