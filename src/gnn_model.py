"""
Graph Neural Network for cross-asset OFI flow propagation.

Models the market as a directed graph:
  - Nodes = assets (Nifty, BankNifty, HDFCBANK, RELIANCE, INFY)
  - Edges = OFI influence channels (learnable weights)
  - Node features = OFI at multiple horizons

The GNN learns which assets' order flow propagates to others,
capturing the cross-asset information channel that OLS/Ridge cannot.

Architecture:
  Node features: [OFI_1, OFI_5, OFI_15, OFI_30, OFI_60] per asset
  → GCN layers (message passing across asset graph)
  → Readout (target node embedding)
  → Dense → predicted return
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from typing import Optional

# Try importing torch_geometric; fall back gracefully
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ── Dataset ───────────────────────────────────────────────────────────

def build_graph_data(
    ofi_df: pd.DataFrame,
    y: pd.Series,
    tickers: list[str],
    horizons: list[int],
    target_node_idx: int = 1,
) -> list:
    """
    Convert OFI DataFrame into a list of PyG Data objects (one per timestep).

    Each graph:
      - num_nodes = len(tickers)
      - node features = OFI at each horizon [N, H]
      - edges = fully connected (every asset can influence every other)
      - y = forward return of target asset
    """
    if not HAS_PYG:
        raise ImportError("torch_geometric required. Install: pip install torch-geometric")

    n_assets = len(tickers)

    # Build fully connected edge index (all-to-all)
    src = []
    dst = []
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Align ofi_df and y on a common index BEFORE iterating.
    # `y` (from prepare_dataset) has NaN rows already dropped, so its index
    # is a strict subset of ofi_df.index.  Without this join, positional
    # indexing (iloc) goes out-of-bounds because y is shorter than ofi_df.
    ofi_cols_needed = [f"{t}_ofi_{h}" for t in tickers for h in horizons
                       if f"{t}_ofi_{h}" in ofi_df.columns]
    joined = ofi_df[ofi_cols_needed].join(y.rename("__y__"), how="inner").dropna()

    # Extract node features per timestep
    graphs = []
    for t_idx in range(len(joined)):
        node_feats = []
        skip = False
        for ticker in tickers:
            feats = []
            for h in horizons:
                col = f"{ticker}_ofi_{h}"
                val = joined[col].iloc[t_idx]
                if np.isnan(val):
                    skip = True
                    break
                feats.append(val)
            if skip:
                break
            node_feats.append(feats)

        if skip:
            continue

        x = torch.tensor(node_feats, dtype=torch.float32)  # [N, H]
        target = torch.tensor(joined["__y__"].iloc[t_idx], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, y=target)
        data.target_node = target_node_idx
        graphs.append(data)

    return graphs


# ── Model ─────────────────────────────────────────────────────────────

class CrossAssetGNN(nn.Module):
    """
    GNN that learns cross-asset OFI flow propagation.

    Uses Graph Attention Network (GAT) layers so the model can
    learn different attention weights for each edge — telling us
    which asset's OFI most influences the target.
    """

    def __init__(
        self,
        in_channels: int = 5,     # OFI horizons per asset
        hidden_channels: int = 32,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric required")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GAT layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Subsequent layers
        for _ in range(n_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))

        # Prediction head (from target node embedding)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)

        # Extract target node embeddings
        # For batched graphs, we need to find the right target node per graph
        n_nodes_per_graph = 5  # number of assets
        target_idx = data.target_node if hasattr(data, "target_node") else 1

        if isinstance(target_idx, int):
            # Single graph or same target for all
            batch_size = batch.max().item() + 1
            indices = torch.arange(batch_size, device=x.device) * n_nodes_per_graph + target_idx
            target_embeds = x[indices]
        else:
            target_embeds = x[target_idx]

        return self.head(target_embeds).squeeze(-1)

    def get_edge_attention(self, data) -> dict:
        """Extract attention weights from GAT layers for interpretability."""
        x, edge_index = data.x, data.edge_index
        attention_weights = {}

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x, (edge_idx, alpha) = conv(x, edge_index, return_attention_weights=True)
            x = norm(x)
            x = F.gelu(x)
            attention_weights[f"layer_{i}"] = {
                "edge_index": edge_idx.detach().cpu(),
                "attention": alpha.detach().cpu(),
            }

        return attention_weights


# ── Training ──────────────────────────────────────────────────────────

class GNNTrainer:
    """Training loop for the GNN with early stopping."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        n_layers: int = 2,
        heads: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 128,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CrossAssetGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            heads=heads,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.history = {"train_loss": [], "val_loss": [], "val_ic": []}

    def fit(self, graphs: list, val_frac: float = 0.2):
        """Train on list of PyG Data objects (time-series split)."""
        n = len(graphs)
        n_val = int(n * val_frac)
        train_graphs = graphs[: n - n_val]
        val_graphs = graphs[n - n_val:]

        from torch_geometric.loader import DataLoader as PyGLoader
        train_loader = PyGLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader = PyGLoader(val_graphs, batch_size=self.batch_size, shuffle=False)

        best_val_ic = -np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                loss = self.criterion(pred, batch.y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            val_loss, val_ic = self._evaluate(val_loader)
            self.history["train_loss"].append(np.mean(train_losses))
            self.history["val_loss"].append(val_loss)
            self.history["val_ic"].append(val_ic)

            if val_ic > best_val_ic:
                best_val_ic = val_ic
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        preds, trues, losses = [], [], []
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
            losses.append(self.criterion(pred, batch.y).item())
            preds.extend(pred.cpu().numpy())
            trues.extend(batch.y.cpu().numpy())
        ic = spearmanr(trues, preds)[0] if len(trues) > 2 else 0.0
        return np.mean(losses), ic

    @torch.no_grad()
    def predict(self, graphs: list) -> np.ndarray:
        from torch_geometric.loader import DataLoader as PyGLoader
        loader = PyGLoader(graphs, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        for batch in loader:
            batch = batch.to(self.device)
            preds.extend(self.model(batch).cpu().numpy())
        return np.array(preds)
