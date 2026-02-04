import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import random

# ===============================
# CONFIGURATION
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 30
LATENT_DIM = 16
BETA = 4.0            # Beta-VAE parameter
LR = 1e-3
NORMAL_DIGIT = 0     # Digit considered "normal"

# ===============================
# DATA LOADING & ANOMALY SETUP
# ===============================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

# Use only one digit as "normal" for training
train_idx = [i for i, (_, y) in enumerate(train_dataset) if y == NORMAL_DIGIT]
train_subset = Subset(train_dataset, train_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# AUTOENCODER (AE) BASELINE
# ===============================
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon.view(-1, 1, 28, 28)

# ===============================
# VARIATIONAL AUTOENCODER (VAE)
# ===============================
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon.view(-1, 1, 28, 28), mu, logvar

# ===============================
# LOSS FUNCTIONS
# ===============================
def ae_loss(recon, x):
    return nn.functional.mse_loss(recon, x, reduction="mean")

def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl

# ===============================
# TRAINING FUNCTIONS
# ===============================
def train_ae(model, loader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, _ in loader:
            x = x.to(DEVICE)
            recon = model(x)
            loss = ae_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"AE Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss / len(loader):.4f}")

def train_vae(model, loader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, _ in loader:
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, BETA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"VAE Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss / len(loader):.4f}")

# ===============================
# ANOMALY SCORING
# ===============================
def compute_scores_ae(model, loader):
    model.eval()
    scores, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            recon = model(x)
            mse = ((x - recon) ** 2).mean(dim=[1, 2, 3])

            scores.extend(mse.cpu().numpy())
            labels.extend((y != NORMAL_DIGIT).numpy())

    return np.array(scores), np.array(labels)

def compute_scores_vae(model, loader):
    model.eval()
    scores, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            recon, _, _ = model(x)
            mse = ((x - recon) ** 2).mean(dim=[1, 2, 3])

            scores.extend(mse.cpu().numpy())
            labels.extend((y != NORMAL_DIGIT).numpy())

    return np.array(scores), np.array(labels)

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":

    ae = AutoEncoder(LATENT_DIM).to(DEVICE)
    vae = VAE(LATENT_DIM).to(DEVICE)

    print("\nTraining AutoEncoder...")
    train_ae(ae, train_loader)

    print("\nTraining Variational AutoEncoder...")
    train_vae(vae, train_loader)

    # Anomaly detection evaluation
    ae_scores, labels = compute_scores_ae(ae, test_loader)
    vae_scores, _ = compute_scores_vae(vae, test_loader)

    ae_auc = roc_auc_score(labels, ae_scores)
    vae_auc = roc_auc_score(labels, vae_scores)

    ae_f1 = f1_score(labels, ae_scores > np.percentile(ae_scores, 95))
    vae_f1 = f1_score(labels, vae_scores > np.percentile(vae_scores, 95))

    print("\n===== ANOMALY DETECTION RESULTS =====")
    print(f"AE  AUC-ROC : {ae_auc:.4f}")
    print(f"VAE AUC-ROC : {vae_auc:.4f}")
    print(f"AE  F1-Score: {ae_f1:.4f}")
    print(f"VAE F1-Score: {vae_f1:.4f}")
