import tools_vae as tvae
import torch
import vae_models as mds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CSV_PATH = "/home/chiara/Energy/Data/Train_2m_72.csv"  # ← Replace this with your actual file\\
 # ← Replace this with your actual file
MODEL_PATH = "/home/chiara/Energy/SAVED_MODEL/VAE"  # ← Replace this with your actual file
SAVE_CSV = "/home/chiara/Energy/PRED/VAE_predictions.csv"

# train the model
while True:
    scaler, ganpath = tvae.train_models(
                    input_dim = 72,
                    hidden_dim = 10,
                    path_data =  CSV_PATH,
                    saving_path =  MODEL_PATH,
                    lr = 0.0002,
                    batch_size = 64,
                    epochs = 2000,
                    save_control = False,
                    model_typle = '30m',
                    )
    break

# Load the model
vae = mds.VAE_30m(input_shape=72, forecast_shape=72, latent_dim=10).to(DEVICE)  # Move the model to the correct device





'''
# Load the state_dict and remove "module." prefix
state_dict = torch.load(ganpath, map_location=DEVICE)  # Ensure the state_dict is loaded on the correct device
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
vae.load_state_dict(new_state_dict)
# Original data
or_data = np.array(pd.read_csv(CSV_PATH, header=0, sep=',', decimal=","))
input_data = or_data[:, :1550]  # First 1550 columns as input
input_data = torch.tensor(input_data, dtype=torch.float32).to(DEVICE)  # Convert to Float and move to device

# Create an instance of MyDataset to initialize scalers
dataset = tvae.MyDataset(or_data)  # Pass the original data to initialize the scalers
# Generate data using the full forward pass
gen_data, _, _ = vae(input_data)  # Forward pass returns decoded data, mu, and logvar
gen_data = gen_data.detach().cpu().numpy()  # Move to CPU and convert to NumPy

# Use the target scaler for inverse transformation
gen_data = dataset.inverse_transform(scaled_data=gen_data, target=True)  # Use the target scaler for forecasted data

# Plot and save results
for i in range(or_data.shape[0]):
    plt.plot(or_data[i, -240:], c='r', alpha=0.1)  # Plot actual last 240 columns
    plt.plot(gen_data[i, :], c='b', alpha=0.1)  # Plot generated 240 columns
plt.plot(or_data[i, -240:], c='r', alpha=0.1, label='original')
plt.plot(gen_data[i, :], c='b', alpha=0.1, label='generated')
plt.legend()
plt.show()
plt.savefig('/home/chiara/Energy/PLOT/vae.png', dpi=300, bbox_inches='tight')

# Save the generated data
pd.DataFrame(gen_data).to_csv(SAVE_CSV, index=False)'''