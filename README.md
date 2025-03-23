# Quant Factor Strategy

Ce projet implémente une stratégie d'investissement quantitative basée sur l'analyse factorielle. Il utilise des facteurs connus tels que value, momentum, qualité et low volatility pour construire un portefeuille optimisé.

## Objectifs

- Collecter et prétraiter les données financières
- Calculer divers facteurs d'investissement
- Construire et optimiser un portefeuille multi-factoriel
- Backtester la stratégie sur des données historiques
- Visualiser les résultats et analyser les performances

## Structure du projet

```
./
├── data/               # Données brutes et prétraitées
├── notebooks/          # Jupyter notebooks pour l'analyse et la visualisation
├── src/                # Code source Python
│   ├── data/           # Scripts de collecte et prétraitement des données
│   ├── factors/        # Implémentation des facteurs
│   ├── portfolio/      # Construction et optimisation du portefeuille
│   ├── backtest/       # Backtesting de la stratégie
│   └── utils/          # Fonctions utilitaires
└── tests/              # Tests unitaires
```

## Installation

```bash
# Cloner le repo
git clone https://github.com/kyac99/quant-factor-strategy.git
cd quant-factor-strategy

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Facteurs implémentés

### Value
- Price-to-Earnings (P/E)
- Price-to-Book (P/B)
- EV/EBITDA

### Momentum
- Rendement relatif sur 12 mois (excluant le dernier mois)
- Momentum sur 3, 6 et 9 mois

### Qualité
- Return on Equity (ROE)
- Return on Assets (ROA)
- Marge bénéficiaire
- Ratio d'endettement

### Low Volatility
- Volatilité historique sur 1 an
- Beta

## Licence

MIT
