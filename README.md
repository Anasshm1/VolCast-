# VolCast

**Prédiction de la volatilité du S&P 500 — comparaison rigoureuse Baseline naïve vs GARCH(1,1) vs LSTM**

Un modèle complexe (LSTM) bat-il vraiment des méthodes plus simples ? Ce projet répond à cette question par une évaluation statistiquement rigoureuse (test de Diebold-Mariano), plutôt que par une comparaison à l'œil.

## TL;DR — Résultats

| Modèle | MAE (out-of-sample) | Amélioration vs Naïve |
|---|---|---|
| Baseline Naïve | 0.004497 | — |
| **GARCH(1,1)** | **0.004128** | **+8.2 %** |
| LSTM (PyTorch) | 0.004462 | +0.8 % |

**GARCH(1,1) bat le LSTM**, de façon statistiquement significative (Diebold-Mariano, p = 0.000081). Résultat cohérent avec une partie de la littérature : sur de la volatilité journalière avec peu de variables d'entrée, les modèles économétriques classiques restent compétitifs face à des réseaux de neurones simples.

## Pourquoi la volatilité, pas le prix

Le prix d'un actif suit une marche quasi aléatoire — essentiellement imprévisible. La volatilité, elle, présente un phénomène de *clustering* : les jours agités arrivent groupés. C'est ce qui rend le problème réellement apprenable.

## Méthode

1. **Données** — S&P 500 (`^GSPC`), quotidien, 2000-2026, via `yfinance`
2. **Cible** — volatilité réalisée sur 5 jours futurs, jamais la volatilité du jour même
3. **Split temporel strict** — train (2000-2019) / test (2020-2026)
4. **3 modèles** entraînés et évalués sur exactement les mêmes données de test
5. **Comparaison honnête** — test de Diebold-Mariano

Détail complet de la méthodologie : voir [`REPORT.md`](REPORT.md).

## Structure du repo

\`\`\`
VolCast/
├── src/
│   ├── data_prep.py      # téléchargement, rendements, volatilité, split, fenêtres
│   ├── models.py          # baseline naïve, GARCH(1,1), LSTM (PyTorch)
│   └── evaluate.py        # métriques, test de Diebold-Mariano
├── volcast.ipynb          # notebook d'exploration complet, phase par phase
├── run_pipeline.py        # reproduit tout le pipeline en une commande
├── requirements.txt
├── REPORT.md               # rapport détaillé
└── README.md
\`\`\`

## Installation

\`\`\`bash
git clone https://github.com/<ton-user>/VolCast.git
cd VolCast
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

## Utilisation

\`\`\`bash
python run_pipeline.py
\`\`\`

## Stack

Python · yfinance · pandas / numpy · PyTorch · arch (GARCH) · scikit-learn · matplotlib

## Licence

MIT — voir [\`LICENSE\`](LICENSE)