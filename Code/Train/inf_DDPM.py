import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ut import EnergyDatasetFromRows
from torchvision import transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
# --------------------- CONFIG ---------------------

EPOCHS = 2000

res = SummaryWriter(log_dir="/home/chiara/Energy/LOG", flush_secs=5)
# Hyperparameters
INPUT_LEN = 1550
OUTPUT_LEN = 240
BATCH_SIZE = 16

TIMESTEPS = 1000
LR = 1e-4
TRAIN_SPLIT = 0.9 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "/home/chiara/Energy/Data/SET_1790.csv"  # ← Replace this with your actual file
SAVE_CSV = "/home/chiara/Energy/PRED/ddpm_predictions"

# --------------------- DIFFUSION ---------------------


def linear_beta_schedule(timesteps):
    return np.linspace(1e-4, 0.02, timesteps, dtype=np.float32)

class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = torch.tensor(linear_beta_schedule(timesteps)).to(DEVICE)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

# --------------------- MODEL ---------------------

class ConditionalTransformer(nn.Module):
    def __init__(self, input_len, output_len, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.cond_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_noisy, cond):
        x = self.input_proj(x_noisy)
        c = self.cond_proj(cond)
        #c = c.unsqueeze(1)
        combined = torch.cat([c, x], dim=1)
        encoded = self.encoder(combined.permute(1, 0, 2))
        out = self.decoder(encoded[-x.size(1):].permute(1, 0, 2))
        return out


# --------------------- SAMPLING ---------------------

@torch.no_grad()
def sample(model, diffusion, cond):
    model.eval()
    x = torch.randn(cond.size(0), OUTPUT_LEN, 1).to(DEVICE)

    for t in reversed(range(diffusion.timesteps)):
        t_batch = torch.full((cond.size(0),), t, device=DEVICE, dtype=torch.long)
        noise_pred = model(x, cond)
        beta = diffusion.betas[t]
        alpha = diffusion.alphas[t]
        alpha_hat = diffusion.alpha_hat[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred) + torch.sqrt(beta) * noise

    return x.squeeze(-1).cpu().numpy()

# --------------------- MAIN ---------------------

def main(jj):
    
    train_df = EnergyDatasetFromRows(CSV_PATH, input_len=INPUT_LEN, output_len=OUTPUT_LEN, train_split=TRAIN_SPLIT, train=True)
    test = EnergyDatasetFromRows(CSV_PATH, input_len=INPUT_LEN, output_len=OUTPUT_LEN, train_split=TRAIN_SPLIT, train=False)

    print(f"DEVICE: {DEVICE}")
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test)}")
    print(f"Train dataset shape: {train_df[0][0].shape}, {train_df[0][1].shape}")
    print(f"Test dataset shape: {test[0][0].shape}, {test[0][1].shape}")
    
    train_loader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
    

    model = ConditionalTransformer(INPUT_LEN, OUTPUT_LEN).to(DEVICE)
    diffusion = Diffusion(TIMESTEPS)

    model.load_state_dict(torch.load("/home/chiara/Energy/SAVED_MODEL/best_DDP.pt"))

    # Test prediction
    test_inputs = torch.stack([x for x, _ in test]).to(DEVICE).unsqueeze(-1)
    predictions = []

    for i in range(0, len(test_inputs), BATCH_SIZE):
        batch_cond = test_inputs[i:i+BATCH_SIZE]
        batch_pred = sample(model, diffusion, batch_cond)
        predictions.extend(batch_pred)

    df_preds = pd.DataFrame(predictions, columns=[f"Pred_{i}" for i in range(OUTPUT_LEN)])
    # Save predictions and actual values
    actual_values = np.array([y for _, y in test]).reshape(-1, OUTPUT_LEN)
    df_actual = pd.DataFrame(actual_values, columns=[f"Actual_{i}" for i in range(OUTPUT_LEN)])
    df_combined = pd.concat([df_preds, df_actual], axis=1)
    df_combined.to_csv(f'{SAVE_CSV}_{jj}.csv', index=False)
    print(f"Predictions and actual values saved to {SAVE_CSV}")




if __name__ == "__main__":
    for i in range(50):
        main(i)
        print(f"Iteration {i} completed.")

