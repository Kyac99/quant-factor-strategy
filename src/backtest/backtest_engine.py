"""
Module pour le backtesting de stratégies d'investissement factorielles.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Moteur de backtest pour évaluer les performances des stratégies d'investissement.
    """
    
    def __init__(self, initial_capital=10000, transaction_costs=0.001, management_fee=0.0025):
        """
        Initialise le moteur de backtest.
        
        Args:
            initial_capital (float): Capital initial en unités monétaires
            transaction_costs (float): Coûts de transaction en pourcentage de la transaction
            management_fee (float): Frais de gestion annuels en pourcentage
        """
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
        self.management_fee = management_fee
    
    def run_backtest(self, weights, prices, rebalance_freq='M'):
        """
        Exécute un backtest basé sur les poids du portefeuille et les prix historiques.
        
        Args:
            weights (pandas.DataFrame): Poids du portefeuille pour chaque actif et chaque date
            prices (pandas.DataFrame): Prix historiques des actifs
            rebalance_freq (str): Fréquence de rééquilibrage ('D' pour quotidien, 'W' pour hebdomadaire, 'M' pour mensuel)
            
        Returns:
            tuple: (equity_curve, performance_metrics, positions, trades)
        """
        try:
            # S'assurer que les index sont des dates
            weights.index = pd.to_datetime(weights.index)
            prices.index = pd.to_datetime(prices.index)
            
            # Aligner les poids et les prix sur les mêmes colonnes (actifs)
            common_assets = weights.columns.intersection(prices.columns)
            weights = weights[common_assets]
            prices = prices[common_assets]
            
            # Déterminer les dates de rééquilibrage
            if rebalance_freq == 'D':
                rebalance_dates = weights.index
            else:
                # Regrouper par fréquence et prendre la dernière date de chaque période
                rebalance_dates = weights.groupby(pd.Grouper(freq=rebalance_freq)).apply(lambda x: x.index[-1] if not x.empty else None).dropna()
            
            # Initialiser les structures de données pour suivre la performance
            equity_curve = pd.Series(index=prices.index, dtype=float)
            equity_curve.iloc[0] = self.initial_capital
            
            positions = pd.DataFrame(0, index=prices.index, columns=common_assets)
            portfolio_value = pd.Series(index=prices.index, dtype=float)
            portfolio_value.iloc[0] = self.initial_capital
            
            trades = []
            cash = self.initial_capital
            current_weights = pd.Series(0, index=common_assets)
            last_rebalance_price = prices.iloc[0]
            
            # Simuler le trading jour par jour
            for i in range(1, len(prices)):
                date = prices.index[i]
                prev_date = prices.index[i-1]
                
                # Mettre à jour les positions basées sur les changements de prix
                for asset in common_assets:
                    if positions.loc[prev_date, asset] != 0:
                        if np.isnan(prices.loc[date, asset]) or np.isnan(prices.loc[prev_date, asset]):
                            # Gérer les prix manquants en maintenant la position constante
                            positions.loc[date, asset] = positions.loc[prev_date, asset]
                        else:
                            # Mettre à jour les positions en fonction des variations de prix
                            positions.loc[date, asset] = positions.loc[prev_date, asset] * (prices.loc[date, asset] / prices.loc[prev_date, asset])
                
                # Calculer la valeur actuelle du portefeuille
                portfolio_value.loc[date] = cash + sum(positions.loc[date, asset] for asset in common_assets)
                
                # Appliquer les frais de gestion quotidiens
                daily_fee = (self.management_fee / 252) * portfolio_value.loc[date]
                portfolio_value.loc[date] -= daily_fee
                cash -= daily_fee
                
                # Vérifier si c'est une date de rééquilibrage
                if date in rebalance_dates:
                    target_weights = weights.loc[date]
                    
                    # Calculer les poids actuels
                    current_value = portfolio_value.loc[date]
                    current_weights = pd.Series(0, index=common_assets)
                    
                    for asset in common_assets:
                        asset_value = prices.loc[date, asset] * positions.loc[date, asset]
                        current_weights[asset] = asset_value / current_value if current_value > 0 else 0
                    
                    # Calculer les ajustements de position nécessaires
                    for asset in common_assets:
                        target_weight = target_weights[asset]
                        current_weight = current_weights[asset]
                        
                        if not np.isnan(target_weight) and not np.isnan(current_weight):
                            # Calculer la valeur à acheter/vendre
                            target_value = target_weight * current_value
                            current_asset_value = current_weight * current_value
                            trade_value = target_value - current_asset_value
                            
                            if abs(trade_value) > 0:
                                # Calculer le nombre d'unités à acheter/vendre
                                price = prices.loc[date, asset]
                                
                                if not np.isnan(price) and price > 0:
                                    trade_units = trade_value / price
                                    
                                    # Appliquer les coûts de transaction
                                    transaction_cost = abs(trade_value) * self.transaction_costs
                                    cash -= transaction_cost
                                    portfolio_value.loc[date] -= transaction_cost
                                    
                                    # Mettre à jour les positions et le cash
                                    positions.loc[date, asset] += trade_units
                                    cash -= trade_value
                                    
                                    # Enregistrer la transaction
                                    trades.append({
                                        'date': date,
                                        'asset': asset,
                                        'units': trade_units,
                                        'price': price,
                                        'value': trade_value,
                                        'transaction_cost': transaction_cost
                                    })
                    
                    last_rebalance_price = prices.loc[date].copy()
                
                # Mettre à jour la courbe d'équité
                equity_curve.loc[date] = portfolio_value.loc[date]
            
            # Calculer les métriques de performance
            performance_metrics = self._calculate_performance_metrics(equity_curve, prices)
            
            return equity_curve, performance_metrics, positions, pd.DataFrame(trades)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du backtest: {e}")
            return pd.Series(), {}, pd.DataFrame(), pd.DataFrame()
    
    def _calculate_performance_metrics(self, equity_curve, prices):
        """
        Calcule les métriques de performance à partir de la courbe d'équité.
        
        Args:
            equity_curve (pandas.Series): Courbe d'équité du portefeuille
            prices (pandas.DataFrame): Prix historiques des actifs
            
        Returns:
            dict: Dictionnaire des métriques de performance
        """
        try:
            # Calculer les rendements
            returns = equity_curve.pct_change().dropna()
            
            # Métriques de base
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            
            # Calculer les rendements annualisés
            years = len(returns) / 252  # 252 jours de trading par an
            annual_return = (1 + total_return) ** (1 / years) - 1
            
            # Calculer la volatilité annualisée
            volatility = returns.std() * np.sqrt(252)
            
            # Calculer le ratio de Sharpe (en supposant un taux sans risque de 0)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculer le drawdown maximal
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Calculer le ratio de Sortino (en supposant un taux sans risque de 0)
            negative_returns = returns.where(returns < 0, 0)
            downside_deviation = negative_returns.std() * np.sqrt(252)
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calculer le ratio de Calmar
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Calculer la fréquence des gains
            win_rate = len(returns[returns > 0]) / len(returns)
            
            # Comparaison avec le marché (si disponible)
            if 'SPY' in prices.columns or '^GSPC' in prices.columns:
                benchmark_ticker = 'SPY' if 'SPY' in prices.columns else '^GSPC'
                benchmark_returns = prices[benchmark_ticker].pct_change().dropna()
                
                # Aligner les rendements du portefeuille et du benchmark
                common_index = returns.index.intersection(benchmark_returns.index)
                aligned_returns = returns.loc[common_index]
                aligned_benchmark = benchmark_returns.loc[common_index]
                
                # Calculer le beta
                covariance = aligned_returns.cov(aligned_benchmark)
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                # Calculer l'alpha
                benchmark_annual_return = (1 + aligned_benchmark.sum()) ** (252 / len(aligned_benchmark)) - 1
                alpha = annual_return - beta * benchmark_annual_return
            else:
                beta = None
                alpha = None
            
            # Rassembler toutes les métriques
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'beta': beta,
                'alpha': alpha
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques de performance: {e}")
            return {}
    
    @staticmethod
    def compare_strategies(results_dict):
        """
        Compare les performances de plusieurs stratégies.
        
        Args:
            results_dict (dict): Dictionnaire avec les noms des stratégies comme clés et les résultats de backtest comme valeurs
            
        Returns:
            pandas.DataFrame: Tableau comparatif des performances
        """
        try:
            comparison = {}
            
            for strategy_name, results in results_dict.items():
                if len(results) >= 2:
                    equity_curve, metrics, _, _ = results
                    comparison[strategy_name] = metrics
            
            return pd.DataFrame(comparison).T
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison des stratégies: {e}")
            return pd.DataFrame()
