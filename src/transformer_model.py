"""
Cross-Asset Transformer for OFI signal prediction.

Architecture:
  Input: OFI sequences from multiple assets (T timesteps x N assets)
  → Time encoding + asset embedding
  → Transformer encoder (multi-head self-attention)
  → Dense head → predicted return

The attention mechanism learns:
  - Which asset's OFI leads (dynamic lead-lag)
  - When relationships are strongest (time-dependent attention)
  - Nonlinear cross-asset interactions
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from typing import Optional


# ── Dataset ───────────────────────────────────────────────────────────

class OFISequenceDataset(Dataset):
    """
    Converts OFI DataFrame into (sequence, target) pairs for the transformer.

    Each sample: X = [T x N_assets] OFI values, y = forward return.
    """

    def __init__(self, ofi_df: pd.DataFrame, y: pd.Series,
                 tickers: list[str], seq_len: int = 30):
        self.seq_len = seq_len
        # Build [T, N_assets] tensor from 1-min OFI columns
        ofi_cols = [f"{t}_ofi_1" for t in tickers]
        available = [c for c in ofi_cols if c in ofi_df.columns]
        self.n_assets = len(available)

        # Align ofi_df and y on a common index BEFORE converting to numpy.
        # `y` (from prepare_dataset) has NaNs already dropped, so its index
        # is a subset of ofi_df.index. Without this join the two arrays
        # have different lengths and the boolean mask below would fail.
        joined = ofi_df[available].join(y.rename("__y__"), how="inner").dropna()
        ofi_arr = joined[available].values.astype(np.float32)
        y_arr = joined["__y__"].values.astype(np.float32)

        valid = ~(np.isnan(ofi_arr).any(axis=1) | np.isnan(y_arr))
        self.ofi = ofi_arr[valid]
        self.y = y_arr[valid]

    def __len__(self):
        return max(0, len(self.ofi) - self.seq_len)

    def __getitem__(self, idx):
        x = self.ofi[idx: idx + self.seq_len]      # [seq_len, n_assets]
        y = self.y[idx + self.seq_len - 1]          # scalar
        return torch.from_numpy(x), torch.tensor(y)


# ── Model ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal ordering."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CrossAssetTransformer(nn.Module):
    """
    Transformer encoder that learns cross-asset OFI relationships.

    Parameters
    ----------
    n_assets : int — number of input assets
    d_model : int — internal embedding dimension
    n_heads : int — attention heads
    n_layers : int — transformer encoder layers
    seq_len : int — input sequence length
    dropout : float
    """

    def __init__(
        self,
        n_assets: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model

        # Asset embedding: project N_assets → d_model
        self.input_proj = nn.Linear(n_assets, d_model)

        # Learnable asset-type embedding (optional enrichment)
        self.asset_embed = nn.Embedding(n_assets, d_model)

        # Positional encoding for time dimension
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, n_assets]
        returns: [batch, 1] predicted return
        """
        # Project input to d_model dimension
        h = self.input_proj(x)  # [B, T, d_model]

        # Add positional encoding
        h = self.pos_encoder(h)

        # Transformer encoder
        h = self.transformer(h)  # [B, T, d_model]

        # Use last timestep's representation for prediction
        h_last = h[:, -1, :]  # [B, d_model]

        return self.head(h_last).squeeze(-1)  # [B]

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract attention weights for interpretability."""
        weights = []
        h = self.input_proj(x)
        h = self.pos_encoder(h)

        for layer in self.transformer.layers:
            # Get attention weights from self-attention
            attn_out, attn_weights = layer.self_attn(
                h, h, h, need_weights=True, average_attn_weights=False
            )
            weights.append(attn_weights.detach())
            # Continue forward pass
            h = layer(h)

        return weights


# ── Training ──────────────────────────────────────────────────────────

class TransformerTrainer:
    """Training loop with early stopping and metrics tracking."""

    def __init__(
        self,
        n_assets: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CrossAssetTransformer(
            n_assets=n_assets, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, seq_len=seq_len,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": [], "val_ic": []}

    def fit(self, ofi_df: pd.DataFrame, y: pd.Series,
            tickers: list[str], val_frac: float = 0.2):
        """Train the transformer with early stopping on validation IC."""
        dataset = OFISequenceDataset(ofi_df, y, tickers, self.seq_len)
        n = len(dataset)
        n_val = int(n * val_frac)
        n_train = n - n_val

        # Time-series split (no shuffling!)
        train_ds = torch.utils.data.Subset(dataset, range(n_train))
        val_ds = torch.utils.data.Subset(dataset, range(n_train, n))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_val_ic = -np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            # Validate
            val_loss, val_ic = self._evaluate(val_loader)
            avg_train = np.mean(train_losses)

            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(val_loss)
            self.history["val_ic"].append(val_ic)

            # Early stopping
            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)

        return self

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        all_preds, all_true, losses = [], [], []
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(xb)
            losses.append(self.criterion(pred, yb).item())
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

        ic = spearmanr(all_true, all_preds)[0] if len(all_true) > 2 else 0.0
        return np.mean(losses), ic

    @torch.no_grad()
    def predict(self, ofi_df: pd.DataFrame, y: pd.Series,
                tickers: list[str]) -> np.ndarray:
        """Generate predictions for the entire dataset."""
        dataset = OFISequenceDataset(ofi_df, y, tickers, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        for xb, _ in loader:
            xb = xb.to(self.device)
            preds.extend(self.model(xb).cpu().numpy())
        return np.array(preds)

    @torch.no_grad()
    def get_attention_map(self, ofi_df: pd.DataFrame, y: pd.Series,
                          tickers: list[str], n_samples: int = 100) -> np.ndarray:
        """Average attention weights over samples for interpretability."""
        dataset = OFISequenceDataset(ofi_df, y, tickers, self.seq_len)
        loader = DataLoader(dataset, batch_size=min(n_samples, len(dataset)), shuffle=False)
        self.model.eval()

        xb, _ = next(iter(loader))
        xb = xb.to(self.device)
        weights = self.model.get_attention_weights(xb)

        # Average over batch and heads: [n_layers, seq_len, seq_len]
        avg_weights = []
        for w in weights:
            avg_weights.append(w.mean(dim=(0, 1)).cpu().numpy())
        return np.stack(avg_weights)
