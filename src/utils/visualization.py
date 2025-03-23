"""
Module pour la visualisation des résultats de la stratégie d'investissement.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)

class VisualizationTools:
    """
    Classe pour la visualisation des résultats et des analyses des stratégies d'investissement.
    """
    
    @staticmethod
    def set_plotting_style():
        """
        Configure le style des graphiques pour une présentation cohérente.
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Format des dates sur l'axe x
        plt.rcParams['axes.formatter.useoffset'] = False
        
        # Configuration de seaborn
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
    
    @staticmethod
    def plot_equity_curve(equity_curve, benchmark=None, title="Performance du Portefeuille", figsize=(12, 8)):
        """
        Trace la courbe d'équité du portefeuille avec une comparaison optionnelle avec un benchmark.
        
        Args:
            equity_curve (pandas.Series): Courbe d'équité du portefeuille
            benchmark (pandas.Series, optional): Courbe d'équité du benchmark
            title (str): Titre du graphique
            figsize (tuple): Dimensions du graphique
            
        Returns:
            matplotlib.figure.Figure: Objet Figure créé
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Normaliser les courbes à 100 au début
            norm_equity = equity_curve / equity_curve.iloc[0] * 100
            norm_equity.plot(ax=ax, linewidth=2, label="Portefeuille", color='#0072B2')
            
            if benchmark is not None:
                # S'assurer que le benchmark et l'equity curve ont le même index
                common_index = norm_equity.index.intersection(benchmark.index)
                norm_benchmark = benchmark.loc[common_index] / benchmark.loc[common_index].iloc[0] * 100
                norm_benchmark.plot(ax=ax, linewidth=2, linestyle='--', label="Benchmark", color='#D55E00')
            
            # Formater l'axe y pour afficher les pourcentages
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
            
            # Ajouter des annotations
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Croissance du Capital (%)', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Améliorer l'apparence générale
            fig.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du tracé de la courbe d'équité: {e}")
            return plt.figure()
    
    @staticmethod
    def plot_drawdowns(equity_curve, top_n=5, figsize=(12, 8)):
        """
        Trace les drawdowns du portefeuille.
        
        Args:
            equity_curve (pandas.Series): Courbe d'équité du portefeuille
            top_n (int): Nombre de drawdowns à afficher
            figsize (tuple): Dimensions du graphique
            
        Returns:
            matplotlib.figure.Figure: Objet Figure créé
        """
        try:
            # Calculer les drawdowns
            returns = equity_curve.pct_change().dropna()
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max) - 1
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Tracer la série de drawdowns
            drawdown.plot(ax=ax, linewidth=2, color='#0072B2')
            
            # Identifier les top N drawdowns
            drawdown_periods = []
            current_drawdown = 0
            start_date = None
            
            for date, value in drawdown.items():
                if value < 0:
                    if start_date is None:
                        start_date = date
                    current_drawdown = min(current_drawdown, value)
                elif start_date is not None:
                    end_date = date
                    drawdown_periods.append((start_date, end_date, current_drawdown))
                    current_drawdown = 0
                    start_date = None
            
            # Ajouter le dernier drawdown s'il est en cours
            if start_date is not None:
                drawdown_periods.append((start_date, drawdown.index[-1], current_drawdown))
            
            # Trier les drawdowns par ampleur
            drawdown_periods.sort(key=lambda x: x[2])
            top_drawdowns = drawdown_periods[:top_n]
            
            # Ajouter des annotations pour les top N drawdowns
            for i, (start, end, value) in enumerate(top_drawdowns):
                mid_date = start + (end - start) / 2
                ax.annotate(f"{value:.1%}", xy=(mid_date, value / 2), ha='center',
                            color='red', fontsize=12, fontweight='bold')
            
            # Formater l'axe y pour afficher les pourcentages
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
            
            # Ajouter des annotations
            ax.set_title("Drawdowns du Portefeuille", fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Drawdown (%)', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Améliorer l'apparence générale
            fig.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du tracé des drawdowns: {e}")
            return plt.figure()
    
    @staticmethod
    def plot_rolling_metrics(equity_curve, window=252, metrics=['return', 'volatility', 'sharpe'], figsize=(18, 10)):
        """
        Trace les métriques glissantes du portefeuille.
        
        Args:
            equity_curve (pandas.Series): Courbe d'équité du portefeuille
            window (int): Fenêtre glissante en jours
            metrics (list): Liste des métriques à tracer
            figsize (tuple): Dimensions du graphique
            
        Returns:
            matplotlib.figure.Figure: Objet Figure créé
        """
        try:
            # Calculer les rendements
            returns = equity_curve.pct_change().dropna()
            
            # Initialiser les métriques glissantes
            rolling_metrics = pd.DataFrame(index=returns.index)
            
            # Calculer le rendement annualisé glissant
            if 'return' in metrics:
                rolling_return = returns.rolling(window=window).apply(
                    lambda x: (1 + x).prod() ** (252 / len(x)) - 1
                )
                rolling_metrics['Rendement Annualisé'] = rolling_return
            
            # Calculer la volatilité annualisée glissante
            if 'volatility' in metrics:
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                rolling_metrics['Volatilité Annualisée'] = rolling_vol
            
            # Calculer le ratio de Sharpe glissant
            if 'sharpe' in metrics:
                rolling_sharpe = returns.rolling(window=window).apply(
                    lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
                )
                rolling_metrics['Ratio de Sharpe'] = rolling_sharpe
            
            # Calculer le drawdown maximal glissant
            if 'max_drawdown' in metrics:
                def calc_max_drawdown(returns_window):
                    cum_ret = (1 + returns_window).cumprod()
                    running_max = cum_ret.cummax()
                    drawdown = (cum_ret / running_max) - 1
                    return drawdown.min()
                
                rolling_max_drawdown = returns.rolling(window=window).apply(calc_max_drawdown)
                rolling_metrics['Drawdown Maximal'] = rolling_max_drawdown
            
            # Tracer les métriques
            fig, axes = plt.subplots(len(rolling_metrics.columns), 1, figsize=figsize, sharex=True)
            
            # S'assurer que axes est toujours une liste
            if len(rolling_metrics.columns) == 1:
                axes = [axes]
            
            for i, column in enumerate(rolling_metrics.columns):
                rolling_metrics[column].plot(ax=axes[i], linewidth=2, color=f'C{i}')
                
                # Formater l'axe y pour les métriques en pourcentage
                if column in ['Rendement Annualisé', 'Volatilité Annualisée', 'Drawdown Maximal']:
                    axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
                
                axes[i].set_title(column, fontsize=14)
                axes[i].grid(True, alpha=0.3)
            
            # Ajouter des annotations
            fig.suptitle(f"Métriques Glissantes ({window} jours)", fontsize=16)
            axes[-1].set_xlabel('Date', fontsize=14)
            
            # Améliorer l'apparence générale
            fig.tight_layout()
            fig.subplots_adjust(top=0.92)
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du tracé des métriques glissantes: {e}")
            return plt.figure()
    
    @staticmethod
    def plot_monthly_returns_heatmap(equity_curve, figsize=(12, 8)):
        """
        Trace une heatmap des rendements mensuels.
        
        Args:
            equity_curve (pandas.Series): Courbe d'équité du portefeuille
            figsize (tuple): Dimensions du graphique
            
        Returns:
            matplotlib.figure.Figure: Objet Figure créé
        """
        try:
            # Calculer les rendements quotidiens
            returns = equity_curve.pct_change().dropna()
            
            # Convertir l'index en datetime si nécessaire
            returns.index = pd.to_datetime(returns.index)
            
            # Regrouper par année et mois et calculer les rendements composés
            monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Convertir en DataFrame avec les années comme index et les mois comme colonnes
            monthly_returns_matrix = pd.DataFrame()
            
            for (year, month), value in monthly_returns.items():
                if year not in monthly_returns_matrix.index:
                    monthly_returns_matrix.loc[year, month] = value
                else:
                    monthly_returns_matrix.at[year, month] = value
            
            # Trier les index et les colonnes
            monthly_returns_matrix = monthly_returns_matrix.sort_index(axis=0)
            monthly_returns_matrix = monthly_returns_matrix.reindex(columns=range(1, 13))
            
            # Remplacer les noms des colonnes par les noms des mois
            month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
            monthly_returns_matrix.columns = month_names
            
            # Créer la heatmap
            fig, ax = plt.subplots(figsize=figsize)
            
            # Définir la palette de couleurs (vert pour positif, rouge pour négatif)
            cmap = sns.diverging_palette(10, 133, as_cmap=True)
            
            # Tracer la heatmap
            sns.heatmap(monthly_returns_matrix, cmap=cmap, annot=True, fmt='.1%',
                       linewidths=0.5, center=0, cbar_kws={'shrink': 0.8}, ax=ax)
            
            # Ajouter des annotations
            ax.set_title('Rendements Mensuels (%)', fontsize=16)
            ax.set_ylabel('Année', fontsize=14)
            
            # Améliorer l'apparence générale
            fig.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du tracé de la heatmap des rendements mensuels: {e}")
            return plt.figure()
    
    @staticmethod
    def plot_factor_contribution(factor_scores, weights, returns, figsize=(12, 8)):
        """
        Trace la contribution de chaque facteur à la performance.
        
        Args:
            factor_scores (dict): Dictionnaire de DataFrames des scores par facteur
            weights (pandas.DataFrame): Poids du portefeuille pour chaque actif
            returns (pandas.DataFrame): Rendements des actifs
            figsize (tuple): Dimensions du graphique
            
        Returns:
            matplotlib.figure.Figure: Objet Figure créé
        """
        try:
            # Calculer les contributions par facteur
            contributions = pd.DataFrame(index=returns.index, columns=factor_scores.keys())
            
            for factor_name, scores in factor_scores.items():
                # Calculer la corrélation entre les scores des facteurs et les rendements
                factor_returns = pd.Series(index=returns.index, dtype=float)
                
                for date in returns.index:
                    if date in scores.index and date in weights.index:
                        # Calculer le rendement pondéré par les scores pour ce facteur
                        weighted_return = 0
                        total_score = 0
                        
                        for asset in returns.columns:
                            if asset in scores.columns and asset in weights.columns:
                                if not np.isnan(scores.loc[date, asset]) and not np.isnan(returns.loc[date, asset]):
                                    score = scores.loc[date, asset]
                                    asset_return = returns.loc[date, asset]
                                    asset_weight = weights.loc[date, asset]
                                    
                                    weighted_return += score * asset_return * asset_weight
                                    total_score += abs(score) * asset_weight
                        
                        if total_score > 0:
                            factor_returns.loc[date] = weighted_return / total_score
                
                # Calculer la contribution cumulée
                contributions[factor_name] = factor_returns.cumsum()
            
            # Tracer les contributions
            fig, ax = plt.subplots(figsize=figsize)
            
            # Tracer chaque facteur
            for factor_name in contributions.columns:
                contributions[factor_name].plot(ax=ax, linewidth=2, label=factor_name)
            
            # Ajouter des annotations
            ax.set_title("Contribution Cumulée des Facteurs", fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Contribution Cumulée', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Améliorer l'apparence générale
            fig.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du tracé des contributions des facteurs: {e}")
            return plt.figure()
    
    @staticmethod
    def plot_factor_exposure(weights, factor_scores, date=None, top_n=10, figsize=(12, 8)):
        """
        Trace l'exposition aux facteurs à une date donnée ou la moyenne sur la période.
        
        Args:
            weights (pandas.DataFrame): Poids du portefeuille pour chaque actif
            factor_scores (dict): Dictionnaire de DataFrames des scores par facteur
            date (str, optional): Date spécifique pour l'analyse (None pour la moyenne)
            top_n (int): Nombre d'actifs à afficher
            figsize (tuple): Dimensions du graphique
            
        Returns:
            matplotlib.figure.Figure: Objet Figure créé
        """
        try:
            # Préparer les données pour l'exposition aux facteurs
            exposure = pd.DataFrame(columns=factor_scores.keys())
            
            if date is not None and date in weights.index:
                # Exposition à une date spécifique
                current_weights = weights.loc[date]
                top_assets = current_weights.nlargest(top_n).index
                
                for asset in top_assets:
                    asset_weight = current_weights[asset]
                    
                    if asset_weight > 0:
                        asset_exposure = {}
                        
                        for factor_name, scores in factor_scores.items():
                            if date in scores.index and asset in scores.columns:
                                asset_exposure[factor_name] = scores.loc[date, asset]
                        
                        exposure.loc[asset] = asset_exposure
            else:
                # Exposition moyenne sur toute la période
                average_weights = weights.mean()
                top_assets = average_weights.nlargest(top_n).index
                
                for asset in top_assets:
                    avg_weight = average_weights[asset]
                    
                    if avg_weight > 0:
                        asset_exposure = {}
                        
                        for factor_name, scores in factor_scores.items():
                            if asset in scores.columns:
                                asset_exposure[factor_name] = scores[asset].mean()
                        
                        exposure.loc[asset] = asset_exposure
            
            # Normaliser les expositions
            for column in exposure.columns:
                exposure[column] = (exposure[column] - exposure[column].mean()) / exposure[column].std()
            
            # Tracer l'exposition
            fig, ax = plt.subplots(figsize=figsize)
            
            # Création du heatmap
            sns.heatmap(exposure, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                       linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
            
            # Ajouter des annotations
            title = f"Exposition aux Facteurs ({date})" if date else "Exposition Moyenne aux Facteurs"
            ax.set_title(title, fontsize=16)
            ax.set_ylabel('Actif', fontsize=14)
            ax.set_xlabel('Facteur', fontsize=14)
            
            # Améliorer l'apparence générale
            fig.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du tracé de l'exposition aux facteurs: {e}")
            return plt.figure()
    
    @staticmethod
    def plot_performance_metrics(metrics, benchmark_metrics=None, figsize=(12, 8)):
        """
        Trace un tableau comparatif des métriques de performance.
        
        Args:
            metrics (dict): Métriques de performance de la stratégie
            benchmark_metrics (dict, optional): Métriques de performance du benchmark
            figsize (tuple): Dimensions du graphique
            
        Returns:
            matplotlib.figure.Figure: Objet Figure créé
        """
        try:
            # Préparer les données pour la comparaison
            metrics_df = pd.DataFrame(index=['Stratégie'])
            
            # Ajouter les métriques clés
            key_metrics = [
                ('total_return', 'Rendement Total'),
                ('annual_return', 'Rendement Annualisé'),
                ('volatility', 'Volatilité'),
                ('sharpe_ratio', 'Ratio de Sharpe'),
                ('max_drawdown', 'Drawdown Maximal'),
                ('sortino_ratio', 'Ratio de Sortino'),
                ('calmar_ratio', 'Ratio de Calmar'),
                ('win_rate', 'Taux de Gain')
            ]
            
            for key, name in key_metrics:
                if key in metrics:
                    metrics_df[name] = [metrics[key]]
            
            # Ajouter le benchmark si disponible
            if benchmark_metrics is not None:
                benchmark_row = {}
                
                for key, name in key_metrics:
                    if key in benchmark_metrics:
                        benchmark_row[name] = benchmark_metrics[key]
                
                metrics_df.loc['Benchmark'] = benchmark_row
            
            # Formater les données pour l'affichage
            formatted_df = metrics_df.copy()
            
            # Formatage spécifique pour chaque type de métrique
            for name in formatted_df.columns:
                if name in ['Rendement Total', 'Rendement Annualisé', 'Volatilité', 'Drawdown Maximal', 'Taux de Gain']:
                    formatted_df[name] = formatted_df[name].apply(lambda x: f"{x:.2%}")
                else:
                    formatted_df[name] = formatted_df[name].apply(lambda x: f"{x:.2f}")
            
            # Créer la figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Cacher les axes
            ax.axis('off')
            
            # Créer le tableau
            table = ax.table(
                cellText=formatted_df.values,
                rowLabels=formatted_df.index,
                colLabels=formatted_df.columns,
                cellLoc='center',
                loc='center',
                cellColours=np.full_like(formatted_df.values, 'white', dtype=object)
            )
            
            # Styliser le tableau
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # En-tête de colonne
                    cell.set_text_props(fontproperties=dict(weight='bold'))
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(color='white')
                elif j == -1:  # En-tête de ligne
                    cell.set_text_props(fontproperties=dict(weight='bold'))
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(color='white')
                else:
                    if i == 1 and j >= 0:  # Ligne Stratégie
                        cell.set_facecolor('#E6F0FF')
                    elif i == 2 and j >= 0:  # Ligne Benchmark
                        cell.set_facecolor('#F5F5F5')
            
            # Ajouter un titre
            ax.set_title('Métriques de Performance', fontsize=16)
            
            # Améliorer l'apparence générale
            fig.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du tracé des métriques de performance: {e}")
            return plt.figure()
