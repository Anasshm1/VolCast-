"""
data_prep.py — Téléchargement et préparation des données pour VolCast.

Pipeline : prix bruts -> rendements log -> volatilité réalisée (passée et cible)
-> split temporel -> fenêtres glissantes pour le LSTM.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(ticker: str = "^GSPC", start: str = "2000-01-01") -> pd.Series:
    """Télécharge les prix de clôture ajustés pour un ticker donné."""
    df = yf.download(ticker, start=start, auto_adjust=True)
    return df["Close"].squeeze()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Rendements journaliers logarithmiques."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_rolling_volatility(returns: pd.Series, window: int = 5) -> pd.Series:
    """Écart-type glissant des rendements (volatilité réalisée)."""
    return returns.rolling(window=window).std()


def build_target(volatility: pd.Series, horizon: int = 5, name: str = "Target_Vol_Future_5d") -> pd.Series:
    """
    Construit la cible : la volatilité des `horizon` jours FUTURS, ramenée
    sur la ligne du jour présent via shift(-horizon).
    """
    target = volatility.shift(-horizon)
    target.name = name
    return target


def build_clean_dataset(
    prices: pd.Series,
    vol_window: int = 5,
    target_horizon: int = 5,
) -> pd.DataFrame:
    """
    Construit le dataset complet et nettoyé :
    Log-Return, Vol_Past_{vol_window}d, Target_Vol_Future_{target_horizon}d.
    """
    log_returns = compute_log_returns(prices)
    vol_past = compute_rolling_volatility(log_returns, window=vol_window)
    target = build_target(vol_past, horizon=target_horizon,
                           name=f"Target_Vol_Future_{target_horizon}d")

    df = pd.DataFrame({
        "Log-Return": log_returns,
        f"Vol_Past_{vol_window}d": vol_past,
        target.name: target,
    })
    return df.dropna()


def split_train_test(df: pd.DataFrame, cutoff_date: str = "2020-01-01"):
    """Split TEMPOREL strict — jamais aléatoire pour une série temporelle."""
    train = df[df.index < cutoff_date]
    test = df[df.index >= cutoff_date]
    return train, test


def create_windows(df: pd.DataFrame, feature_col: str, target_col: str, window_size: int = 20):
    """
    Construit des fenêtres glissantes (X, y) pour le LSTM.
    À appeler SÉPARÉMENT sur train et sur test pour éviter toute fuite
    de données à la frontière temporelle.
    """
    features = df[feature_col].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(features[i - window_size:i])
        y.append(targets[i])

    return np.array(X), np.array(y)