import gan_tool as tg
import torch 
import torch.nn as nn
import GAN as mds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


CSV_PATH = "/home/chiara/Energy/Data/Train_2m_72.csv"  # ← Replace this with your actual file
MODEL_PATH = "/home/chiara/Energy/SAVED_MODEL/GAN"  # ← Replace this with your actual file
SAVE_CSV = "/home/chiara/Energy/PRED/gan_predictions.csv"
#  train the model
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
 
'''# load the model
#ganpaht = f'/home/chiara/Energy/SAVED_MODEL/GAN/models/generator__0_20250505-210340.pt' # The path of stored models
gen = mds.Generator('60m', 1550)
state_dict = torch.load(ganpaht)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
gen.load_state_dict(new_state_dict)

# original data

or_data = np.array(pd.read_csv(CSV_PATH, header=0, sep=',', decimal=","))
gen_data = gen(torch.rand(or_data.shape[0], 1550)).detach().numpy()  # Input 1550 hours
gen_data = scaler.inverse_transform(gen_data)

for i in range(or_data.shape[0]):
    plt.plot(or_data[i,:], c='r', alpha=0.1)
    # plt.plot(gen_data[i,:], c='b', alpha=0.1)
plt.plot(or_data[i,:], c='r', alpha=0.1, label='original')
# plt.plot(gen_data[i,:], c='b', alpha=0.1, label='generated')
plt.legend()
plt.show()
plt.savefig('/home/chiara/Energy/PLOT/gan.png', dpi=300, bbox_inches='tight')

# save the generated data

pd.DataFrame(gen_data).to_csv(SAVE_CSV, index=False)
'''