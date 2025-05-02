import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ut import EnergyDatasetFromRows

# --------------------- CONFIG ---------------------

INPUT_LEN = 1550
OUTPUT_LEN = 240
BATCH_SIZE = 16
EPOCHS = 10
TIMESTEPS = 1000
LR = 1e-4
TRAIN_SPLIT = 0.9 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "/home/chiara/Energy/Data/SET_1790.csv"  # ← Replace this with your actual file
SAVE_CSV = "ddpm_predictions.csv"

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

# --------------------- TRAINING ---------------------

def train(model, diffusion, dataloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE).unsqueeze(-1), y.to(DEVICE).unsqueeze(-1)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=DEVICE).long()
            noise = torch.randn_like(y)
            y_noisy = diffusion.add_noise(y, t, noise)
            pred = model(y_noisy, x)
            loss = mse(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

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

def main():
    
    train_df = EnergyDatasetFromRows(CSV_PATH, input_len=INPUT_LEN, output_len=OUTPUT_LEN, train_split=TRAIN_SPLIT, train=True)
    test = EnergyDatasetFromRows(CSV_PATH, input_len=INPUT_LEN, output_len=OUTPUT_LEN, train_split=TRAIN_SPLIT, train=False)

    print(f"DEVICE: {DEVICE}")
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test)}")
    print(f"Train dataset shape: {train_df[0][0].shape}, {train_df[0][1].shape}")
    print(f"Test dataset shape: {test[0][0].shape}, {test[0][1].shape}")
#
    train_loader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
    

    model = ConditionalTransformer(INPUT_LEN, OUTPUT_LEN).to(DEVICE)
    diffusion = Diffusion(TIMESTEPS)

    train(model, diffusion, train_loader)

    # Test prediction
    test_inputs = torch.stack([x for x, _ in test]).to(DEVICE).unsqueeze(-1)
    predictions = []

    for i in range(0, len(test_inputs), BATCH_SIZE):
        batch_cond = test_inputs[i:i+BATCH_SIZE]
        batch_pred = sample(model, diffusion, batch_cond)
        predictions.extend(batch_pred)

    # Save predictions
    flat_predictions = np.array(predictions).reshape(-1, OUTPUT_LEN)
    df_preds = pd.DataFrame(flat_predictions)
    df_preds.to_csv(SAVE_CSV, index=False)
    print(f"Predictions saved to {SAVE_CSV}")

if __name__ == "__main__":
    main()
