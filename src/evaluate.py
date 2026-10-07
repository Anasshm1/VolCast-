"""
evaluate.py — Métriques et test statistique pour comparer les modèles honnêtement.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE et RMSE pour une paire (vérité, prédiction)."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def compare_models(y_true: np.ndarray, predictions: dict) -> pd.DataFrame:
    """
    predictions : dict {nom_du_modele: array_de_predictions}
    Retourne un DataFrame trié par MAE, avec l'amélioration % vs le premier modèle.
    """
    results = {name: compute_metrics(y_true, preds) for name, preds in predictions.items()}
    df_results = pd.DataFrame(results).T.sort_values("MAE")

    baseline_mae = list(results.values())[0]["MAE"]
    df_results["Amélioration MAE vs 1er modèle (%)"] = (
        (baseline_mae - df_results["MAE"]) / baseline_mae * 100
    )
    return df_results


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1):
    """
    Test de Diebold-Mariano.
    H0 : les deux modèles ont la même précision de prédiction.
    e1, e2 : erreurs de prédiction (y_true - y_pred) des deux modèles, mêmes dates.

    Retourne (statistique DM, p-value). DM < 0 -> modèle 1 plus précis en moyenne.
    p_value < 0.05 -> différence statistiquement significative.
    """
    d = np.abs(e1) - np.abs(e2)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value