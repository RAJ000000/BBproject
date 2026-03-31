import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. SEED
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# 2. LOAD + PREPROCESS
#    condition = subject + sessionIndex
# =========================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    if "subject" not in df.columns:
        raise ValueError("Missing required column: subject")
    if "sessionIndex" not in df.columns:
        raise ValueError("Missing required column: sessionIndex")

    if "rep" in df.columns:
        df = df.drop(columns=["rep"])

    df = df.dropna().reset_index(drop=True)

    exclude_cols = ["subject", "sessionIndex"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if len(feature_cols) == 0:
        raise ValueError("No timing feature columns found")

    X = df[feature_cols].values

    cond_df = df[["subject", "sessionIndex"]].copy()

    try:
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(sparse=False)

    C = encoder.fit_transform(cond_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data shape:", df.shape)
    print("Feature shape:", X_scaled.shape)
    print("Condition shape:", C.shape)

    return df, X_scaled, C, scaler, encoder, feature_cols


# =========================================================
# 3. SPLIT BY USER AND SESSION
# =========================================================
def split_data_by_user_session(df, X, C, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    rng = np.random.RandomState(random_state)

    train_idx = []
    val_idx = []
    test_idx = []

    groups = df.groupby(["subject", "sessionIndex"]).indices

    for _, idx in groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(round(n * train_size))
        n_val = int(round(n * val_size))

        if n_train + n_val > n:
            n_val = n - n_train

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    X_train, C_train = X[train_idx], C[train_idx]
    X_val, C_val = X[val_idx], C[val_idx]
    X_test, C_test = X[test_idx], C[test_idx]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print("Train:", X_train.shape, C_train.shape)
    print("Val  :", X_val.shape, C_val.shape)
    print("Test :", X_test.shape, C_test.shape)

    return df_train, df_val, df_test, X_train, X_val, X_test, C_train, C_val, C_test


# =========================================================
# 4. DATASET + LOADERS
# =========================================================
class CMUDataset(Dataset):
    def __init__(self, X, C):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx]


def create_loaders(X_train, X_val, C_train, C_val, batch_size=64):
    train_loader = DataLoader(CMUDataset(X_train, C_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CMUDataset(X_val, C_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# =========================================================
# 5. MODEL
# =========================================================
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=48, dropout=0.15):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


def build_model(input_dim, cond_dim, latent_dim=48, dropout=0.15):
    return CVAE(input_dim, cond_dim, latent_dim=latent_dim, dropout=dropout)


# =========================================================
# 6. LOSS
# =========================================================
def loss_fn(recon, x, mu, logvar, beta=0.0005):
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


# =========================================================
# 7. TRAIN
# =========================================================
def train(model, train_loader, val_loader, epochs=150, lr=0.0005, beta=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train = 0.0

        for x, c in train_loader:
            optimizer.zero_grad()

            recon, mu, logvar = model(x, c)
            loss, _, _ = loss_fn(recon, x, mu, logvar, beta=beta)

            loss.backward()
            optimizer.step()

            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val = 0.0

        with torch.no_grad():
            for x, c in val_loader:
                recon, mu, logvar = model(x, c)
                loss, _, _ = loss_fn(recon, x, mu, logvar, beta=beta)
                total_val += loss.item()

        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1}/{epochs} | Train {avg_train:.4f} | Val {avg_val:.4f}")

    return model, train_losses, val_losses


# =========================================================
# 8. SAVE / LOAD MODEL
# =========================================================
def save_model(model, filename="cvae_model.pth"):
    torch.save(model.state_dict(), filename)
    print("Saved model to:", os.path.abspath(filename))


def load_model(model, filename="cvae_model.pth"):
    model.load_state_dict(torch.load(filename, map_location="cpu"))
    model.eval()
    print("Loaded model from:", os.path.abspath(filename))
    return model


# =========================================================
# 9. GENERATE SAME SHAPE AS REAL DATA
#    matches subject/session counts exactly
# =========================================================
def generate_same_shape(model, df, scaler, encoder, feature_cols):
    model.eval()

    counts = (
        df.groupby(["subject", "sessionIndex"])
        .size()
        .reset_index(name="count")
    )

    all_blocks = []

    with torch.no_grad():
        for _, row in counts.iterrows():
            subject = row["subject"]
            session = row["sessionIndex"]
            count = int(row["count"])

            cond_df = pd.DataFrame([{
                "subject": subject,
                "sessionIndex": session
            }])

            c = encoder.transform(cond_df)
            c_tensor = torch.tensor(c, dtype=torch.float32)

            samples = []

            for _ in range(count):
                z = torch.randn(1, model.latent_dim)
                sample = model.decode(z, c_tensor).numpy().flatten()
                samples.append(sample)

            samples = np.array(samples)
            samples_original = scaler.inverse_transform(samples)

            meta = pd.DataFrame({
                "subject": [subject] * count,
                "sessionIndex": [session] * count
            })

            feat_df = pd.DataFrame(samples_original, columns=feature_cols)

            block_df = pd.concat(
                [meta.reset_index(drop=True), feat_df.reset_index(drop=True)],
                axis=1
            )

            all_blocks.append(block_df)

    synthetic_df = pd.concat(all_blocks, ignore_index=True)

    synthetic_df["subject"] = synthetic_df["subject"].astype(str)

    if pd.api.types.is_numeric_dtype(df["sessionIndex"]):
        synthetic_df["sessionIndex"] = pd.to_numeric(synthetic_df["sessionIndex"])
    else:
        synthetic_df["sessionIndex"] = synthetic_df["sessionIndex"].astype(str)

    print("Real shape      :", df.shape)
    print("Synthetic shape :", synthetic_df.shape)

    return synthetic_df


def save_dataset(df, filename="synthetic_cmu.csv"):
    df.to_csv(filename, index=False)
    print("Saved dataset to:", os.path.abspath(filename))


# =========================================================
# 10. PLOTS
# =========================================================
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.show()


def plot_pca(real_df, syn_df, feature_cols):
    pca = PCA(n_components=2)

    real = pca.fit_transform(real_df[feature_cols])
    syn = pca.transform(syn_df[feature_cols])

    plt.figure(figsize=(8, 6))
    plt.scatter(real[:, 0], real[:, 1], alpha=0.3, label="Real")
    plt.scatter(syn[:, 0], syn[:, 1], alpha=0.3, label="Synthetic")
    plt.title("PCA Comparison")
    plt.legend()
    plt.show()


def plot_feature_hist(real_df, syn_df, feature_name):
    plt.figure(figsize=(8, 5))
    plt.hist(real_df[feature_name], bins=30, alpha=0.5, label="Real")
    plt.hist(syn_df[feature_name], bins=30, alpha=0.5, label="Synthetic")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature_name}")
    plt.legend()
    plt.show()


# =========================================================
# 11. STATS
# =========================================================
def compare_stats(real_df, syn_df, feature_cols, num_features=5):
    print("REAL MEANS")
    print(real_df[feature_cols].mean().head(num_features))

    print("\nSYNTHETIC MEANS")
    print(syn_df[feature_cols].mean().head(num_features))

    print("\nREAL STDS")
    print(real_df[feature_cols].std().head(num_features))

    print("\nSYNTHETIC STDS")
    print(syn_df[feature_cols].std().head(num_features))


# =========================================================
# 12. SVM TEST
# =========================================================
def svm_utility_test(real_df, syn_df, feature_cols):
    X_real = real_df[feature_cols].values
    y_real = real_df["subject"].values

    X_syn = syn_df[feature_cols].values
    y_syn = syn_df["subject"].values

    # fixed real test set
    from sklearn.model_selection import train_test_split
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
    )

    svm_real = SVC(kernel="rbf")
    svm_real.fit(Xr_train, yr_train)
    pred_real = svm_real.predict(Xr_test)
    acc_real = accuracy_score(yr_test, pred_real)

    svm_syn = SVC(kernel="rbf")
    svm_syn.fit(X_syn, y_syn)
    pred_syn = svm_syn.predict(Xr_test)
    acc_syn = accuracy_score(yr_test, pred_syn)

    print("SVM accuracy (train real -> test real):", round(acc_real, 4))
    print("SVM accuracy (train synthetic -> test real):", round(acc_syn, 4))

    return acc_real, acc_syn