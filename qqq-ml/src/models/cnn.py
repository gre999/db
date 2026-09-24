"""Week 11 (phase 5, week 1): 1D-CNN for the minute-sequence tasks.

CPU-only PyTorch (config/week11_dl.toml [cnn]): ``Conv1d(4->16,k=3) ->
ReLU -> Conv1d(16->32,k=3) -> ReLU -> AdaptiveAvgPool1d(1) -> Dropout(0.3)
-> Linear(32->1)``, 1809 parameters. Class-weighted BCEWithLogitsLoss
(pos_weight from the fit window, same convention as
``src.models.filter.make_xgboost``'s scale_pos_weight), Adam lr=1e-3,
batch_size=32, early-stopped on validation-window loss (patience=10).

Trains 5 fixed seeds per fold (config/week11_dl.toml [cnn.multi_seed],
amended before running: a single seed's init/data-order variance on
~500-700 rows could be the same order of magnitude as the CNN-vs-baseline
AUC gap being measured, while every baseline is deterministic - comparing
one CNN seed to a deterministic baseline would not be a fair test). The
5 seeds' predicted probabilities are averaged; that averaged probability
is what gets compared to a baseline. Individual per-seed AUCs are also
reported (see :func:`fit_predict_fold`'s ``test_auc_per_seed`` output) so
seed variance stays visible.

Week 11 only proves this on ONE fold with three sanity checks
(:func:`overfit_tiny_batch_check`, :func:`shuffled_label_check`, and a
direct AUC comparison against the best baseline on that same fold) -
week 12 is the full walk-forward.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from src import sequences as SEQ
from src.validation import WalkForwardSplit

log = logging.getLogger(__name__)

SEEDS = (0, 1, 2, 3, 4)
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3


class SmallCNN(nn.Module):
    def __init__(self, n_channels: int = SEQ.N_CHANNELS, c1: int = 16, c2: int = 32,
                kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(n_channels, c1, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (batch, timesteps, channels) - the same layout
        ``src.sequences`` tensors use; transposed internally for Conv1d's
        (batch, channels, timesteps) convention."""
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x).squeeze(-1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    w = n_neg / n_pos if n_pos > 0 else 1.0
    return torch.tensor([w], dtype=torch.float32)


def train_one_seed(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray,
                   y_val: np.ndarray, seed: int, max_epochs: int = MAX_EPOCHS,
                   patience: int = PATIENCE, batch_size: int = BATCH_SIZE,
                   lr: float = LR, dropout: float = 0.3) -> SmallCNN:
    """Fits one seed, early-stopped on validation-window loss. Deterministic
    given ``seed`` (torch.manual_seed + use_deterministic_algorithms)."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = SmallCNN(dropout=dropout)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y_fit))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xfit_t = torch.tensor(X_fit, dtype=torch.float32)
    yfit_t = torch.tensor(y_fit, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32)
    yval_t = torch.tensor(y_val, dtype=torch.float32)

    n = len(Xfit_t)
    gen = torch.Generator().manual_seed(seed)
    best_val_loss, best_state, no_improve = float("inf"), None, 0

    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xfit_t[idx]), yfit_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_proba(model: SmallCNN, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def fit_predict_fold(tensor: np.ndarray, days: pd.DatetimeIndex, y: pd.Series, f,
                     validation_years: int = 1, seeds=SEEDS, **train_kwargs) -> dict:
    """One fold, ``len(seeds)`` independently-trained CNNs - see the
    module docstring. Returns a dict with the test-fold predictions
    DataFrame (averaged-probability, ``model="cnn"`` - same schema as
    every other model this project saves), the per-seed test AUCs, and
    the fitted models (for the sanity checks to reuse)."""
    train_dates = days[f.train_idx]
    test_dates = days[f.test_idx]
    validation_start = f.test_start - pd.DateOffset(years=validation_years)
    fit_dates = train_dates[train_dates < validation_start]
    val_dates = train_dates[train_dates >= validation_start]

    std_tensor, _, _ = SEQ.standardize(tensor, days, fit_dates)
    fit_mask, val_mask, test_mask = (days.isin(fit_dates), days.isin(val_dates),
                                     days.isin(test_dates))
    X_fit, y_fit = std_tensor[fit_mask], y.to_numpy()[fit_mask]
    X_val, y_val = std_tensor[val_mask], y.to_numpy()[val_mask]
    X_test, y_test = std_tensor[test_mask], y.to_numpy()[test_mask]

    test_probas, models = [], []
    for seed in seeds:
        model = train_one_seed(X_fit, y_fit, X_val, y_val, seed, **train_kwargs)
        models.append(model)
        test_probas.append(predict_proba(model, X_test))
    test_probas = np.stack(test_probas)          # (n_seeds, n_test)
    mean_proba = test_probas.mean(axis=0)

    test_auc_per_seed = [float(roc_auc_score(y_test, p)) for p in test_probas] \
        if len(np.unique(y_test)) > 1 else [np.nan] * len(seeds)
    mean_auc = float(roc_auc_score(y_test, mean_proba)) if len(np.unique(y_test)) > 1 else np.nan

    preds = pd.DataFrame({"date": test_dates, "fold": f.number, "model": "cnn",
                         "target": "cnn_task", "y_true": y_test.astype(int),
                         "y_pred_proba": mean_proba})
    return {"preds": preds, "test_auc_mean_proba": mean_auc,
           "test_auc_per_seed": test_auc_per_seed, "models": models,
           "X_fit": X_fit, "y_fit": y_fit, "X_val": X_val, "y_val": y_val,
           "X_test": X_test, "y_test": y_test}


# --------------------------------------------------------------------------
# Sanity checks (config/week11_dl.toml [cnn.sanity_checks])
# --------------------------------------------------------------------------
def overfit_tiny_batch_check(X_fit: np.ndarray, y_fit: np.ndarray, n: int = 32,
                             seed: int = 0, max_epochs: int = 300) -> dict:
    """A fixed n-example subset of the fit window must reach >=95% training
    accuracy with no early stopping and no dropout - confirms the model
    can actually learn, not just that the training loop runs."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_fit), size=min(n, len(X_fit)), replace=False)
    Xs, ys = X_fit[idx], y_fit[idx]
    if len(np.unique(ys)) < 2:
        # degenerate draw - resample deterministically until both classes appear
        pos = np.where(y_fit == 1)[0]
        neg = np.where(y_fit == 0)[0]
        half = n // 2
        idx = np.concatenate([rng.choice(pos, min(half, len(pos)), replace=False),
                              rng.choice(neg, min(n - half, len(neg)), replace=False)])
        Xs, ys = X_fit[idx], y_fit[idx]

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = SmallCNN(dropout=0.0)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xs_t = torch.tensor(Xs, dtype=torch.float32)
    ys_t = torch.tensor(ys, dtype=torch.float32)

    model.train()
    for _ in range(max_epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xs_t), ys_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(Xs_t)) >= 0.5).numpy().astype(int)
    acc = float(accuracy_score(ys, pred))
    return {"n": len(Xs), "train_accuracy": acc, "passes": acc >= 0.95}


def shuffled_label_check(X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray,
                         y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                         seed: int = 0, **train_kwargs) -> dict:
    """Fit on the fit window with the LABEL COLUMN SHUFFLED (features
    untouched) - test-fold AUC must be within [0.45, 0.55] of chance.
    Materially above 0.5 means something is leaking (a shuffled label has,
    by construction, no real relationship to the inputs)."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y_fit)
    model = train_one_seed(X_fit, y_shuffled, X_val, y_val, seed, **train_kwargs)
    proba = predict_proba(model, X_test)
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else np.nan
    return {"test_auc": auc, "passes": bool(0.45 <= auc <= 0.55) if not np.isnan(auc) else False}


def first_fold_with_enough_data(days: pd.DatetimeIndex, validation_years: int = 1,
                                min_fit: int = 100, min_val: int = 100,
                                splitter: WalkForwardSplit = WalkForwardSplit()):
    """Week 11's scope is one fold - the earliest one with enough fit and
    validation data (config/week11_dl.toml [split] week11_scope)."""
    for f in splitter.folds(days):
        train_dates = days[f.train_idx]
        validation_start = f.test_start - pd.DateOffset(years=validation_years)
        n_fit = int((train_dates < validation_start).sum())
        n_val = int((train_dates >= validation_start).sum())
        if n_fit >= min_fit and n_val >= min_val and len(f.test_idx) > 0:
            return f
    raise ValueError("no fold has enough fit+validation data")
