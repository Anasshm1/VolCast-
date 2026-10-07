"""
models.py — Les 3 modèles comparés dans VolCast :
baseline naïve, GARCH(1,1), LSTM (PyTorch).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from arch import arch_model


# ---------------------------------------------------------------------------
# 1. Baseline naïve
# ---------------------------------------------------------------------------

def naive_predict(df: pd.DataFrame, past_vol_col: str = "Vol_Past_5d") -> np.ndarray:
    """Prédiction naïve : volatilité de demain = volatilité d'aujourd'hui."""
    return df[past_vol_col].values


# ---------------------------------------------------------------------------
# 2. GARCH(1,1)
# ---------------------------------------------------------------------------

def fit_garch(log_returns: pd.Series, mean: str = "Zero", p: int = 1, q: int = 1):
    """
    Entraîne un GARCH(p,q) sur les rendements (en %, convertis en interne).
    Retourne l'objet résultat (res_garch) : res_garch.conditional_volatility / 100
    donne la volatilité conditionnelle à l'échelle décimale d'origine.
    """
    returns_pct = log_returns * 100
    model = arch_model(returns_pct, mean=mean, vol="Garch", p=p, q=q, dist="normal")
    return model.fit(disp="off")


def fit_garch_fixed_on_full_series(train_returns: pd.Series, full_returns: pd.Series,
                                    mean: str = "Zero", p: int = 1, q: int = 1):
    """
    Pratique recommandée pour une évaluation honnête :
    1. estime les paramètres GARCH uniquement sur `train_returns`
    2. applique ces paramètres FIXES à `full_returns` (train+test) pour obtenir
       la volatilité conditionnelle jour par jour, sans ré-estimation sur le test.
    La récursion GARCH à la date t n'utilise que les rendements <= t : pas de fuite.
    """
    res_train = fit_garch(train_returns, mean=mean, p=p, q=q)

    full_returns_pct = full_returns * 100
    model_full = arch_model(full_returns_pct, mean=mean, vol="Garch", p=p, q=q, dist="normal")
    res_fixed = model_full.fix(res_train.params)
    return res_fixed


# ---------------------------------------------------------------------------
# 3. LSTM
# ---------------------------------------------------------------------------

class VolLSTM(nn.Module):
    """LSTM à une couche : lit une séquence de rendements, prédit une volatilité."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)


def train_lstm(
    model: VolLSTM,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    verbose: bool = True,
):
    """Boucle d'entraînement standard (Adam + MSE)."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if y_train_t.dim() == 1:
        y_train_t = y_train_t.unsqueeze(-1)

    num_samples = X_train_t.shape[0]
    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            idx = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= num_samples
        loss_history.append(epoch_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Époque [{epoch:2d}/{epochs}] - Loss (MSE) : {epoch_loss:.6f}")

    return model, loss_history


def predict_lstm(model: VolLSTM, X_t: torch.Tensor) -> np.ndarray:
    """Prédiction en mode évaluation (pas de gradient)."""
    model.eval()
    with torch.no_grad():
        preds = model(X_t)
    return preds.squeeze().numpy()