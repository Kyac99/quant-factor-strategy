"""
Script principal pour exécuter la stratégie d'investissement factorielle.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.data.data_loader import DataLoader
from src.factors.factor_model import FactorModel
from src.portfolio.portfolio_builder import PortfolioBuilder
from src.backtest.backtest_engine import BacktestEngine
from src.utils.visualization import VisualizationTools

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("strategy.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Fonction principale qui exécute la stratégie complète.
    """
    try:
        logger.info("Démarrage de la stratégie d'investissement factorielle")
        
        # Configuration des paramètres
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Liste des tickers à utiliser (exemple avec des actions du S&P 500)
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'BAC', 'MA', 'XOM', 'DIS', 'NVDA', 'PYPL', 'ADBE',
            'CRM', 'CMCSA', 'NFLX', 'PFE', 'ABT', 'KO', 'PEP', 'TMO', 'CSCO', 'ACN'
        ]
        
        # Charger les données
        logger.info("Chargement des données")
        data_loader = DataLoader(start_date=start_date, end_date=end_date)
        
        # Récupérer les données de prix
        price_data = data_loader.get_stock_data(tickers)
        
        # Récupérer les données fondamentales
        fundamentals = data_loader.get_fundamentals(tickers)
        
        # Récupérer les données de l'indice
        market_data = data_loader.get_market_index()
        
        # Combiner les prix de clôture ajustés
        prices_df = data_loader.combine_prices(price_data)
        
        # Convertir market_data en Series
        market_series = market_data['Adj Close']
        
        # Calcul des rendements
        returns_df = prices_df.pct_change().dropna()
        
        # Initialiser le modèle factoriel
        logger.info("Calcul des scores factoriels")
        factor_model = FactorModel()
        
        # Calculer les scores factoriels
        value_score = factor_model.calculate_value_score(prices_df, fundamentals)
        momentum_score = factor_model.calculate_momentum_score(prices_df)
        quality_score = factor_model.calculate_quality_score(fundamentals)
        low_vol_score = factor_model.calculate_low_volatility_score(prices_df, market_series)
        
        # Calculer le score combiné
        combined_score = factor_model.calculate_combined_score(
            prices_df, fundamentals, market_series
        )
        
        # Classer les actifs
        ranks = factor_model.rank_assets(combined_score)
        
        # Sélectionner les meilleurs actifs
        selected_assets = factor_model.select_top_assets(ranks, top_n=10)
        
        # Construction du portefeuille
        logger.info("Construction du portefeuille")
        portfolio_builder = PortfolioBuilder()
        
        # Comparer différentes méthodes de construction
        equal_weights = portfolio_builder.equal_weight(selected_assets)
        score_weights = portfolio_builder.score_weighted(selected_assets, combined_score)
        min_var_weights = portfolio_builder.minimum_variance(selected_assets, returns_df)
        max_sharpe_weights = portfolio_builder.maximum_sharpe_ratio(selected_assets, returns_df)
        risk_parity_weights = portfolio_builder.risk_parity(selected_assets, returns_df)
        
        # Appliquer des contraintes aux poids
        constrained_weights = portfolio_builder.apply_constraints(max_sharpe_weights)
        
        # Backtest des différentes stratégies
        logger.info("Exécution des backtests")
        backtest_engine = BacktestEngine()
        
        backtest_results = {}
        strategies = {
            'Equal Weight': equal_weights,
            'Score Weighted': score_weights,
            'Minimum Variance': min_var_weights,
            'Maximum Sharpe': max_sharpe_weights,
            'Risk Parity': risk_parity_weights
        }
        
        for name, weights in strategies.items():
            logger.info(f"Backtest de la stratégie: {name}")
            results = backtest_engine.run_backtest(weights, prices_df)
            backtest_results[name] = results
        
        # Comparer les performances
        comparison = backtest_engine.compare_strategies(backtest_results)
        logger.info("Comparaison des stratégies:\n%s", comparison)
        
        # Visualisation des résultats
        logger.info("Génération des visualisations")
        VisualizationTools.set_plotting_style()
        
        # Créer un dossier pour les graphiques s'il n'existe pas
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Tracer la courbe d'équité pour chaque stratégie
        equity_curves = {name: results[0] for name, results in backtest_results.items()}
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for name, curve in equity_curves.items():
            # Normaliser la courbe à 100 au début
            normalized_curve = curve / curve.iloc[0] * 100
            normalized_curve.plot(ax=ax, label=name, linewidth=2)
        
        # Normaliser la courbe de l'indice aussi
        benchmark_curve = market_series.loc[equity_curves['Equal Weight'].index[0]:equity_curves['Equal Weight'].index[-1]]
        normalized_benchmark = benchmark_curve / benchmark_curve.iloc[0] * 100
        normalized_benchmark.plot(ax=ax, label='Benchmark', linestyle='--', color='black', linewidth=2)
        
        ax.set_title('Performance des Stratégies', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Performance (%)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('plots/equity_curves.png')
        
        # Tracer les drawdowns pour la meilleure stratégie
        best_strategy = comparison.iloc[comparison['annual_return'].argmax()].name
        best_equity_curve = equity_curves[best_strategy]
        
        fig = VisualizationTools.plot_drawdowns(best_equity_curve)
        plt.savefig('plots/drawdowns.png')
        
        # Tracer les rendements mensuels
        fig = VisualizationTools.plot_monthly_returns_heatmap(best_equity_curve)
        plt.savefig('plots/monthly_returns.png')
        
        # Tracer les métriques glissantes
        fig = VisualizationTools.plot_rolling_metrics(best_equity_curve)
        plt.savefig('plots/rolling_metrics.png')
        
        # Tracer l'exposition aux facteurs
        factor_scores = {
            'Value': value_score,
            'Momentum': momentum_score,
            'Quality': quality_score,
            'Low Volatility': low_vol_score
        }
        
        fig = VisualizationTools.plot_factor_exposure(
            strategies[best_strategy], factor_scores, date=prices_df.index[-1]
        )
        plt.savefig('plots/factor_exposure.png')
        
        # Tracer la contribution des facteurs
        fig = VisualizationTools.plot_factor_contribution(
            factor_scores, strategies[best_strategy], returns_df
        )
        plt.savefig('plots/factor_contribution.png')
        
        # Tracer le tableau des métriques de performance
        best_metrics = backtest_results[best_strategy][1]
        benchmark_returns = market_series.pct_change().dropna()
        benchmark_equity = (1 + benchmark_returns).cumprod() * backtest_results[best_strategy][0].iloc[0]
        
        # Calculer les métriques pour le benchmark
        benchmark_metrics = backtest_engine._calculate_performance_metrics(benchmark_equity, prices_df)
        
        fig = VisualizationTools.plot_performance_metrics(best_metrics, benchmark_metrics)
        plt.savefig('plots/performance_metrics.png')
        
        logger.info(f"Visualisations sauvegardées dans le dossier 'plots'")
        logger.info(f"La meilleure stratégie est '{best_strategy}'")
        
        # Afficher un résumé des résultats
        logger.info("\n--- Résumé des performances ---")
        for strategy, (equity, metrics, _, _) in backtest_results.items():
            logger.info(f"{strategy}:")
            logger.info(f"  - Rendement annualisé: {metrics['annual_return']:.2%}")
            logger.info(f"  - Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  - Drawdown maximal: {metrics['max_drawdown']:.2%}")
        
        logger.info("Stratégie d'investissement factorielle terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur dans l'exécution de la stratégie: {e}", exc_info=True)

if __name__ == "__main__":
    main()
