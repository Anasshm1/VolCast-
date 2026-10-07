# VolCast — Prédiction de la volatilité du S&P 500

Premier projet de machine learning appliqué à la finance : comparer honnêtement un LSTM à des méthodes de référence pour la prédiction de volatilité.

## Objectif

Prédire la **volatilité réalisée** du S&P 500 sur les 5 jours de bourse suivants, et déterminer si un réseau de neurones récurrent (LSTM) apporte une vraie valeur ajoutée par rapport à une méthode naïve et à un modèle statistique classique (GARCH), via une comparaison rigoureuse — pas juste "faire marcher" un LSTM.

**Pourquoi la volatilité, pas le prix :** le prix suit une marche quasi aléatoire, essentiellement imprévisible. La volatilité présente un phénomène de *clustering* (les jours agités arrivent groupés), ce qui la rend réellement apprenable.

## Données

- **Source :** S&P 500 (`^GSPC`), via `yfinance`
- **Période :** 2000-01-10 à 2026-10-05
- **Fréquence :** quotidienne (clôture)
- **Taille après nettoyage :** 6719 lignes

## Méthode

1. Rendements logarithmiques journaliers (`Log-Return`)
2. Volatilité réalisée glissante sur 5 jours, passée (`Vol_Past_5d`) et future (`Target_Vol_Future_5d`, la cible à prédire)
3. **Split temporel strict** (jamais aléatoire) : train avant le 2020-01-01, test après
   - Train : 5026 jours (2000-01-10 → 2019-12-31)
   - Test : 1693 jours (2020-01-02 → 2026-09-28), incluant le choc COVID
4. Fenêtres glissantes de 20 jours pour le LSTM, construites séparément sur train et test (aucune fuite à la frontière)

## Modèles comparés

| Modèle | Description |
|---|---|
| Baseline naïve | "volatilité de demain = volatilité d'aujourd'hui" |
| GARCH(1,1) | `mean='Zero'`, estimé sur les rendements journaliers |
| LSTM | `hidden_size=32`, séquences de 20 jours, PyTorch, 50 epochs |

## Résultats — out-of-sample (test, 2020-2026)

| Modèle | MAE | RMSE | Amélioration vs Naïve |
|---|---|---|---|
| Baseline Naïve | 0.004497 | 0.006997 | — |
| **GARCH(1,1)** | **0.004128** | **0.006581** | **+8.2 %** |
| LSTM (PyTorch) | 0.004462 | 0.007263 | +0.8 % |

### Test de Diebold-Mariano (significativité statistique)

| Comparaison | Statistique DM | p-value | Conclusion |
|---|---|---|---|
| GARCH vs LSTM | -3.94 | 0.000081 | GARCH significativement meilleur |
| GARCH vs Baseline | -4.20 | 0.000027 | GARCH significativement meilleur |

## Conclusion

**GARCH(1,1) bat le LSTM sur ce problème**, de façon statistiquement significative. Le LSTM ne s'améliore que marginalement par rapport à la baseline naïve (+0.8 %), bien en-dessous de GARCH (+8.2 %).

Ce résultat n'est pas un échec du projet — c'est exactement le type de conclusion honnête que la méthodologie visait à produire. Il rejoint d'ailleurs une partie de la littérature académique : sur de la volatilité journalière avec peu de variables d'entrée, les modèles économétriques classiques comme GARCH restent souvent compétitifs face à des réseaux de neurones simples, qui ont besoin de plus de données et de features pour exprimer leur avantage théorique.

## Limites

- **GARCH estimé in-sample** (sur tout l'historique 2000-2026) plutôt que strictement sur le train — léger avantage méthodologique en sa faveur par rapport au LSTM, qui lui n'a vu que le train. Une ré-estimation stricte sur train uniquement (voir `guide_phases_5_a_8.md`, section 7.2) donnerait une comparaison encore plus rigoureuse.
- **LSTM univarié** : une seule variable d'entrée (le rendement journalier). Pas de volume, de VIX, ni d'autres indices.
- **Période de test hétérogène** : COVID (2020), reprise, inflation 2022 — régimes très différents moyennés dans un seul score.
- **Architecture LSTM simple** : une seule couche, pas de réglage fin des hyperparamètres (taille cachée, nombre d'epochs, dropout).

## Pistes d'amélioration

- Ré-entraîner GARCH strictement sur le train pour une comparaison 100 % équitable
- Ajouter des features au LSTM (volume, VIX, rendements d'autres indices)
- Essayer GARCH avec rolling re-estimation (ré-ajustement glissant plutôt que paramètres fixes)
- Explorer une architecture plus riche (LSTM multi-couches, GRU, ou Transformer léger)
- Tester d'autres tailles de fenêtre (10, 30, 60 jours) et d'autres horizons de prédiction

## Stack

Python 3.11.9 · yfinance · pandas / numpy · PyTorch · arch (GARCH) · scikit-learn · matplotlib · Jupyter Notebook

## Application réelle

Ce type de modèle de prédiction de volatilité sert au calcul du risque de portefeuille (VaR), au pricing d'options, et à la gestion de position.