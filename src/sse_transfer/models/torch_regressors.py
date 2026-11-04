"""Torch model definitions and a shared early-stopped training loop for the
Phase 1.9 composition-only neural candidates:
  C. regularized shallow MLP (fixed-size tabular input),
  D. pretrained-element-embedding shallow regressor (fixed-size pooled
     embedding input -- same `ShallowMLPRegressor` class as C, smaller),
  E. composition-attention model (CrabNet/Roost-style; variable-length
     element-token input).

CLAUDE.md "Keep model definitions free of data-download logic": every class
and function here consumes only precomputed tensors/arrays built upstream by
`scripts/phase1_9_composition_ood.py`; nothing here reads a dataset, split,
or raw file.

`CompositionAttentionRegressor` is a direct torch reimplementation rather
than a dependency on the `crabnet`/`roost` packages: neither is installed or
pinned in this project (pyproject.toml), and vendoring an external modeling
codebase for a ~360-row dataset is more than "the smallest scientifically
correct change" this phase requires. The architecture follows the same core
idea (Roost, Goodall & Lee 2020): learned per-element identity embeddings,
refined by self-attention over the other elements present in the same
composition, then fraction-weighted pooled into a single composition vector
before a regression head.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

ForwardFn = Callable[[nn.Module, tuple[torch.Tensor, ...]], torch.Tensor]

_LOSS_FNS: dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "huber": nn.SmoothL1Loss(beta=1.0),
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ShallowMLPRegressor(nn.Module):
    """Candidates C and D's regression head: a small stack of
    Linear -> ReLU -> Dropout blocks on a fixed-size input vector."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out


class CompositionAttentionRegressor(nn.Module):
    """Candidate E: element-identity embeddings refined by self-attention
    over the other elements in the same composition, fraction-weighted
    pooled, then a small regression head. See module docstring for the
    CrabNet/Roost-style design rationale."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.element_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.fraction_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(
        self, element_idx: torch.Tensor, fractions: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """`element_idx`/`fractions`/`pad_mask`: shape (batch, max_elements);
        `pad_mask` True at padded positions (nn.TransformerEncoder convention)."""
        x = self.element_embedding(element_idx) + self.fraction_proj(fractions.unsqueeze(-1))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        frac_weights = fractions.masked_fill(pad_mask, 0.0)
        denom = frac_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (x * frac_weights.unsqueeze(-1)).sum(dim=1) / denom
        out: torch.Tensor = self.head(pooled).squeeze(-1)
        return out


@dataclass
class TorchFitResult:
    model: nn.Module
    best_epoch: int
    n_epochs_run: int
    train_loss_history: list[float] = field(default_factory=list)
    val_mae_history: list[float] = field(default_factory=list)
    n_parameters: int = 0
    converged: bool = False  # True if early stopping triggered before max_epochs


def train_with_early_stopping(
    model: nn.Module,
    forward_fn: ForwardFn,
    train_inputs: tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    val_inputs: tuple[torch.Tensor, ...],
    y_val: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    loss: str,
    seed: int,
) -> TorchFitResult:
    """Full-batch Adam training (datasets here are a few hundred rows, so a
    full-batch step per epoch is both simpler and faster than minibatching)
    with early stopping on `val_inputs`/`y_val`'s MAE. The caller is
    responsible for ensuring `val_inputs` never overlaps the fold/rows being
    scored for a reported metric (EXPERIMENT_PROTOCOL.md "Never tune on the
    final test set") -- this function only implements the stopping rule."""
    if loss not in _LOSS_FNS:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {sorted(_LOSS_FNS)}")
    torch.manual_seed(seed)
    loss_fn = _LOSS_FNS[loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_since_improve = 0
    train_history: list[float] = []
    val_history: list[float] = []
    converged = False

    epoch = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = forward_fn(model, train_inputs)
        train_loss = loss_fn(pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = forward_fn(model, val_inputs)
            val_mae = torch.mean(torch.abs(val_pred - y_val)).item()
        train_history.append(float(train_loss.item()))
        val_history.append(val_mae)

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                converged = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TorchFitResult(
        model=model,
        best_epoch=best_epoch,
        n_epochs_run=epoch + 1,
        train_loss_history=train_history,
        val_mae_history=val_history,
        n_parameters=count_parameters(model),
        converged=converged,
    )


def train_fixed_epochs(
    model: nn.Module,
    forward_fn: ForwardFn,
    train_inputs: tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    loss: str,
    seed: int,
) -> nn.Module:
    """No-early-stopping refit for exactly `n_epochs` full-batch steps: used
    to retrain a final model on the *whole* outer-fold training partition
    (train + the inner early-stopping split combined) at the epoch count
    `train_with_early_stopping` already selected on that same partition's
    inner split, so the reported outer-fold metric still never influences
    training."""
    if loss not in _LOSS_FNS:
        raise ValueError(f"Unknown loss {loss!r}; expected one of {sorted(_LOSS_FNS)}")
    torch.manual_seed(seed)
    loss_fn = _LOSS_FNS[loss]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(max(n_epochs, 1)):
        model.train()
        optimizer.zero_grad()
        pred = forward_fn(model, train_inputs)
        loss_value = loss_fn(pred, y_train)
        loss_value.backward()
        optimizer.step()
    model.eval()
    return model


def mlp_forward(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    (x,) = inputs
    result: torch.Tensor = model(x)
    return result


def attention_forward(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    idx, frac, mask = inputs
    result: torch.Tensor = model(idx, frac, mask)
    return result


def to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)
