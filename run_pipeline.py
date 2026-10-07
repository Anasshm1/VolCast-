"""
run_pipeline.py — Reproduit l'ensemble du projet VolCast en une seule commande.

Usage :
    python run_pipeline.py
"""

import torch

from src.data_prep import (
    build_clean_dataset,
    create_windows,
    download_prices,
    split_train_test,
)
from src.evaluate import compare_models, diebold_mariano_test
from src.models import VolLSTM, naive_predict, fit_garch, predict_lstm, train_lstm

WINDOW_SIZE = 20
CUTOFF_DATE = "2020-01-01"


def main():
    print("1. Téléchargement et préparation des données...")
    prices = download_prices("^GSPC", start="2000-01-01")
    df_clean = build_clean_dataset(prices)
    train, test = split_train_test(df_clean, cutoff_date=CUTOFF_DATE)
    print(f"   Train : {len(train)} lignes | Test : {len(test)} lignes")

    print("\n2. Fenêtres glissantes pour le LSTM...")
    X_train, y_train = create_windows(train, "Log-Return", "Target_Vol_Future_5d", WINDOW_SIZE)
    X_test, y_test = create_windows(test, "Log-Return", "Target_Vol_Future_5d", WINDOW_SIZE)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    print("\n3. Entraînement de GARCH(1,1)...")
    res_garch = fit_garch(df_clean["Log-Return"])
    df_clean["Vol_GARCH_1d"] = res_garch.conditional_volatility / 100

    print("\n4. Entraînement du LSTM...")
    model = VolLSTM(input_size=1, hidden_size=32)
    model, _ = train_lstm(model, X_train_t, y_train_t, epochs=50, batch_size=64)

    print("\n5. Évaluation comparée sur le test...")
    df_test_period = df_clean.iloc[-len(y_test):]
    y_true = y_test

    predictions = {
        "Baseline Naïve": df_test_period["Vol_Past_5d"].values,
        "GARCH(1,1)": df_test_period["Vol_GARCH_1d"].values,
        "LSTM (PyTorch)": predict_lstm(model, X_test_t),
    }

    results = compare_models(y_true, predictions)
    print(results.round(6))

    print("\n6. Test de Diebold-Mariano...")
    e_naive = y_true - predictions["Baseline Naïve"]
    e_garch = y_true - predictions["GARCH(1,1)"]
    e_lstm = y_true - predictions["LSTM (PyTorch)"]

    dm_gl, p_gl = diebold_mariano_test(e_garch, e_lstm)
    dm_gn, p_gn = diebold_mariano_test(e_garch, e_naive)
    print(f"GARCH vs LSTM     -> DM={dm_gl:.4f} | p-value={p_gl:.6f}")
    print(f"GARCH vs Baseline -> DM={dm_gn:.4f} | p-value={p_gn:.6f}")


if __name__ == "__main__":
    main()