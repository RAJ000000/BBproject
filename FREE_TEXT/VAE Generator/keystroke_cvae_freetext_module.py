import os
import copy
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
# 2. LOAD + CLEAN + PREPROCESS
# =========================================================
def load_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "participant",
        "session",
        "key1",
        "key2",
        "DU.key1.key1",
        "DD.key1.key2",
        "DU.key1.key2",
        "UD.key1.key2",
        "UU.key1.key2",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    feature_cols = [
        "DU.key1.key1",
        "DD.key1.key2",
        "DU.key1.key2",
        "UD.key1.key2",
        "UU.key1.key2",
    ]

    meta_cols = [c for c in df.columns if c not in feature_cols]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after = len(df)

    print("Rows before cleaning:", before)
    print("Rows after cleaning :", after)
    print("Dropped rows        :", before - after)

    # slightly looser clipping than before, to preserve more variation
    clip_bounds = {}
    for col in feature_cols:
        low = df[col].quantile(0.001)
        high = df[col].quantile(0.999)
        clip_bounds[col] = (low, high)
        df[col] = df[col].clip(lower=low, upper=high)

    X = df[feature_cols].values.astype(np.float32)

    cond_df = df[["participant", "session"]].copy()

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    C = encoder.fit_transform(cond_df).astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # looser scaled clipping to preserve diversity
    X_scaled = np.clip(X_scaled, -6.0, 6.0)

    print("Cleaned data shape:", df.shape)
    print("Feature shape     :", X_scaled.shape)
    print("Condition shape   :", C.shape)

    return df, X_scaled, C, scaler, encoder, feature_cols, meta_cols, clip_bounds


# =========================================================
# 3. SPLIT BY PARTICIPANT + SESSION
# =========================================================
def split_data_by_participant_session(
    df,
    X,
    C,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    random_state=42
):
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    rng = np.random.RandomState(random_state)

    train_idx = []
    val_idx = []
    test_idx = []

    groups = df.groupby(["participant", "session"]).indices

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

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    X_train, C_train = X[train_idx], C[train_idx]
    X_val, C_val = X[val_idx], C[val_idx]
    X_test, C_test = X[test_idx], C[test_idx]

    print("Train:", X_train.shape, C_train.shape)
    print("Val  :", X_val.shape, C_val.shape)
    print("Test :", X_test.shape, C_test.shape)

    return df_train, df_val, df_test, X_train, X_val, X_test, C_train, C_val, C_test


# =========================================================
# 4. DATASET + LOADERS
# =========================================================
class FreeTextDataset(Dataset):
    def __init__(self, X, C):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.C[idx]


def create_loaders(X_train, X_val, C_train, C_val, batch_size=256):
    train_loader = DataLoader(
        FreeTextDataset(X_train, C_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        FreeTextDataset(X_val, C_val),
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader


# =========================================================
# 5. STRONGER CVAE
# =========================================================
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=16, dropout=0.05):
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
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, min=-8.0, max=8.0)
        return mu, logvar

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


def build_model(input_dim, cond_dim, latent_dim=16, dropout=0.05):
    return CVAE(input_dim, cond_dim, latent_dim=latent_dim, dropout=dropout)


# =========================================================
# 6. LOSS
# =========================================================
def loss_fn(recon, x, mu, logvar, beta=0.00005):
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


# =========================================================
# 7. TRAIN
# =========================================================
def train(
    model,
    train_loader,
    val_loader,
    epochs=60,
    lr=0.0003,
    beta=0.00005,
    patience=10
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_train = 0.0
        train_batches = 0

        for x, c in train_loader:
            optimizer.zero_grad()

            recon, mu, logvar = model(x, c)
            loss, _, _ = loss_fn(recon, x, mu, logvar, beta=beta)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train += loss.item()
            train_batches += 1

        avg_train = total_train / max(train_batches, 1)
        train_losses.append(avg_train)

        model.eval()
        total_val = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, c in val_loader:
                recon, mu, logvar = model(x, c)
                loss, _, _ = loss_fn(recon, x, mu, logvar, beta=beta)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_val += loss.item()
                val_batches += 1

        avg_val = total_val / max(val_batches, 1)
        val_losses.append(avg_val)

        print(f"Epoch {epoch + 1}/{epochs} | Train {avg_train:.6f} | Val {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Best model restored. Best val loss:", best_val)

    return model, train_losses, val_losses


# =========================================================
# 8. MOMENT MATCHING
#    fixes weak synthetic variance
# =========================================================
def match_feature_moments(synthetic_df, real_df, feature_cols, clip_bounds):
    synthetic_df = synthetic_df.copy()

    for col in feature_cols:
        real_mean = real_df[col].mean()
        real_std = real_df[col].std()

        syn_mean = synthetic_df[col].mean()
        syn_std = synthetic_df[col].std()

        if syn_std is None or syn_std == 0 or np.isnan(syn_std):
            continue

        synthetic_df[col] = ((synthetic_df[col] - syn_mean) / syn_std) * real_std + real_mean

        low, high = clip_bounds[col]
        synthetic_df[col] = synthetic_df[col].clip(lower=low, upper=high)

    return synthetic_df


# =========================================================
# 9. GENERATE SAME SHAPE + SAME CLEANED COLUMN STRUCTURE
# =========================================================
def generate_same_shape(
    model,
    df,
    scaler,
    encoder,
    feature_cols,
    meta_cols,
    clip_bounds,
    temperature=1.15,
    moment_match=True
):
    model.eval()

    synthetic_parts = []

    with torch.no_grad():
        for (participant, session), group_df in df.groupby(["participant", "session"], sort=False):
            count = len(group_df)

            cond_df = pd.DataFrame([{
                "participant": participant,
                "session": session
            }])

            c = encoder.transform(cond_df).astype(np.float32)
            c_tensor = torch.tensor(c, dtype=torch.float32)

            samples = []

            for _ in range(count):
                z = torch.randn(1, model.latent_dim) * temperature
                sample = model.decode(z, c_tensor).numpy().flatten()
                samples.append(sample)

            samples = np.array(samples, dtype=np.float32)
            samples_original = scaler.inverse_transform(samples)

            for j, col in enumerate(feature_cols):
                low, high = clip_bounds[col]
                samples_original[:, j] = np.clip(samples_original[:, j], low, high)

            meta_df = group_df[meta_cols].reset_index(drop=True).copy()
            feat_df = pd.DataFrame(samples_original, columns=feature_cols)

            synthetic_block = pd.concat([meta_df, feat_df], axis=1)
            synthetic_parts.append(synthetic_block)

    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)

    if moment_match:
        synthetic_df = match_feature_moments(synthetic_df, df, feature_cols, clip_bounds)

    print("Real shape      :", df.shape)
    print("Synthetic shape :", synthetic_df.shape)

    return synthetic_df


def save_dataset(df, filename="synthetic_free_text.csv"):
    df.to_csv(filename, index=False)
    print("Saved dataset to:", os.path.abspath(filename))


# =========================================================
# 10. EVALUATION
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


def compare_stats(real_df, syn_df, feature_cols, num_features=5):
    print("REAL MEANS")
    print(real_df[feature_cols].mean().head(num_features))

    print("\nSYNTHETIC MEANS")
    print(syn_df[feature_cols].mean().head(num_features))

    print("\nREAL STDS")
    print(real_df[feature_cols].std().head(num_features))

    print("\nSYNTHETIC STDS")
    print(syn_df[feature_cols].std().head(num_features))


def plot_feature_hist(real_df, syn_df, feature_name):
    real_vals = pd.to_numeric(real_df[feature_name], errors="coerce").dropna()
    syn_vals = pd.to_numeric(syn_df[feature_name], errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(real_vals, bins=30, alpha=0.5, label="Real")
    plt.hist(syn_vals, bins=30, alpha=0.5, label="Synthetic")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature_name}")
    plt.legend()
    plt.show()


def plot_pca(real_df, syn_df, feature_cols):
    real_num = real_df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    syn_num = syn_df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()

    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_num)
    syn_pca = pca.transform(syn_num)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.3, label="Real")
    plt.scatter(syn_pca[:, 0], syn_pca[:, 1], alpha=0.3, label="Synthetic")
    plt.title("PCA Comparison")
    plt.legend()
    plt.show()


def svm_utility_test_fast(real_df, syn_df, feature_cols, sample_per_participant=2000, random_state=42):
    rng = np.random.RandomState(random_state)

    def stratified_cap(df, label_col, cap):
        parts = []
        for label, group in df.groupby(label_col):
            if len(group) > cap:
                idx = rng.choice(group.index, size=cap, replace=False)
                parts.append(group.loc[idx])
            else:
                parts.append(group)
        return pd.concat(parts).reset_index(drop=True)

    real_small = stratified_cap(real_df, "participant", sample_per_participant)
    syn_small = stratified_cap(syn_df, "participant", sample_per_participant)

    X_real = real_small[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    y_real = real_small.loc[X_real.index, "participant"]

    X_syn = syn_small[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    y_syn = syn_small.loc[X_syn.index, "participant"]

    from sklearn.model_selection import train_test_split

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_real,
        y_real,
        test_size=0.2,
        random_state=random_state,
        stratify=y_real
    )

    svm_real = SVC(kernel="rbf")
    svm_real.fit(Xr_train, yr_train)
    pred_real = svm_real.predict(Xr_test)
    acc_real = accuracy_score(yr_test, pred_real)

    svm_syn = SVC(kernel="rbf")
    svm_syn.fit(X_syn, y_syn)
    pred_syn = svm_syn.predict(Xr_test)
    acc_syn = accuracy_score(yr_test, pred_syn)

    print("Fast SVM accuracy (train real -> test real):", round(acc_real, 4))
    print("Fast SVM accuracy (train synthetic -> test real):", round(acc_syn, 4))

    return acc_real, acc_syn