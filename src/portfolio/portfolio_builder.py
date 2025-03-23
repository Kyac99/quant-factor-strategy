"""
Module pour la construction et l'optimisation de portefeuille.
"""
import logging
import pandas as pd
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class PortfolioBuilder:
    """
    Classe pour construire et optimiser un portefeuille basé sur les scores factoriels.
    """
    
    def __init__(self):
        """
        Initialise le constructeur de portefeuille.
        """
        pass
    
    @staticmethod
    def equal_weight(selected_assets):
        """
        Construit un portefeuille à pondération égale des actifs sélectionnés.
        
        Args:
            selected_assets (pandas.DataFrame): DataFrame booléen indiquant les actifs sélectionnés
            
        Returns:
            pandas.DataFrame: Poids du portefeuille pour chaque actif et chaque date
        """
        try:
            # Initialiser les poids à zéro
            weights = pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
            
            # Pour chaque date, donner un poids égal aux actifs sélectionnés
            for date in weights.index:
                selected_on_date = selected_assets.loc[date]
                n_selected = selected_on_date.sum()
                
                if n_selected > 0:
                    weight = 1.0 / n_selected
                    weights.loc[date, selected_on_date] = weight
                
            return weights
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction du portefeuille à poids égaux: {e}")
            return pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
    
    @staticmethod
    def score_weighted(selected_assets, scores):
        """
        Construit un portefeuille pondéré par les scores factoriels.
        
        Args:
            selected_assets (pandas.DataFrame): DataFrame booléen indiquant les actifs sélectionnés
            scores (pandas.DataFrame): Scores combinés pour chaque actif
            
        Returns:
            pandas.DataFrame: Poids du portefeuille pour chaque actif et chaque date
        """
        try:
            # Initialiser les poids à zéro
            weights = pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
            
            # Pour chaque date, pondérer les actifs sélectionnés par leur score
            for date in weights.index:
                # Filtrer les actifs sélectionnés
                mask = selected_assets.loc[date]
                
                if mask.sum() > 0:
                    # Récupérer les scores des actifs sélectionnés
                    filtered_scores = scores.loc[date].where(mask, 0)
                    
                    # Normaliser les scores pour qu'ils somment à 1
                    sum_scores = filtered_scores.sum()
                    
                    if sum_scores != 0:
                        weights.loc[date] = filtered_scores / sum_scores
                
            return weights
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction du portefeuille pondéré par score: {e}")
            return pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
    
    @staticmethod
    def minimum_variance(selected_assets, returns):
        """
        Construit un portefeuille à variance minimale des actifs sélectionnés.
        
        Args:
            selected_assets (pandas.DataFrame): DataFrame booléen indiquant les actifs sélectionnés
            returns (pandas.DataFrame): Rendements historiques des actifs
            
        Returns:
            pandas.DataFrame: Poids du portefeuille pour chaque actif et chaque date
        """
        try:
            # Initialiser les poids à zéro
            weights = pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
            
            # Pour chaque date, optimiser les poids pour minimiser la variance
            for date in weights.index:
                # Filtrer les actifs sélectionnés
                mask = selected_assets.loc[date]
                selected_tickers = mask[mask].index.tolist()
                
                if len(selected_tickers) > 0:
                    # Calculer la matrice de covariance à partir des rendements historiques
                    # Utiliser une fenêtre de 252 jours (environ 1 an)
                    lookback_end = date
                    lookback_start_idx = max(0, returns.index.get_loc(lookback_end) - 252)
                    lookback_start = returns.index[lookback_start_idx]
                    
                    historical_returns = returns.loc[lookback_start:lookback_end, selected_tickers]
                    cov_matrix = historical_returns.cov()
                    
                    # Définir la fonction objectif pour minimiser la variance
                    def portfolio_variance(w):
                        return np.dot(w.T, np.dot(cov_matrix, w))
                    
                    # Contraintes: les poids doivent sommer à 1 et être positifs
                    n_assets = len(selected_tickers)
                    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                    bounds = tuple((0, 1) for _ in range(n_assets))
                    
                    # Poids initiaux égaux
                    initial_weights = np.ones(n_assets) / n_assets
                    
                    # Optimiser
                    result = minimize(
                        portfolio_variance,
                        initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                    
                    if result.success:
                        optimized_weights = result.x
                        weights.loc[date, selected_tickers] = optimized_weights
                    else:
                        logger.warning(f"Échec de l'optimisation à la date {date}: {result.message}")
                        # Fallback sur poids égaux
                        weights.loc[date, selected_tickers] = 1.0 / n_assets
                
            return weights
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction du portefeuille à variance minimale: {e}")
            return pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
    
    @staticmethod
    def maximum_sharpe_ratio(selected_assets, returns, risk_free_rate=0.0):
        """
        Construit un portefeuille maximisant le ratio de Sharpe des actifs sélectionnés.
        
        Args:
            selected_assets (pandas.DataFrame): DataFrame booléen indiquant les actifs sélectionnés
            returns (pandas.DataFrame): Rendements historiques des actifs
            risk_free_rate (float): Taux sans risque annualisé
            
        Returns:
            pandas.DataFrame: Poids du portefeuille pour chaque actif et chaque date
        """
        try:
            # Initialiser les poids à zéro
            weights = pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
            
            # Convertir le taux sans risque annualisé en taux quotidien
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            
            # Pour chaque date, optimiser les poids pour maximiser le ratio de Sharpe
            for date in weights.index:
                # Filtrer les actifs sélectionnés
                mask = selected_assets.loc[date]
                selected_tickers = mask[mask].index.tolist()
                
                if len(selected_tickers) > 0:
                    # Calculer les rendements moyens et la matrice de covariance
                    lookback_end = date
                    lookback_start_idx = max(0, returns.index.get_loc(lookback_end) - 252)
                    lookback_start = returns.index[lookback_start_idx]
                    
                    historical_returns = returns.loc[lookback_start:lookback_end, selected_tickers]
                    mean_returns = historical_returns.mean() - daily_risk_free
                    cov_matrix = historical_returns.cov()
                    
                    # Définir la fonction objectif pour maximiser le ratio de Sharpe
                    def negative_sharpe_ratio(w):
                        portfolio_return = np.sum(mean_returns * w) * 252  # Annualiser
                        portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) * np.sqrt(252)  # Annualiser
                        
                        if portfolio_volatility == 0:
                            return 0
                        
                        return -portfolio_return / portfolio_volatility
                    
                    # Contraintes: les poids doivent sommer à 1 et être positifs
                    n_assets = len(selected_tickers)
                    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                    bounds = tuple((0, 1) for _ in range(n_assets))
                    
                    # Poids initiaux égaux
                    initial_weights = np.ones(n_assets) / n_assets
                    
                    # Optimiser
                    result = minimize(
                        negative_sharpe_ratio,
                        initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                    
                    if result.success:
                        optimized_weights = result.x
                        weights.loc[date, selected_tickers] = optimized_weights
                    else:
                        logger.warning(f"Échec de l'optimisation à la date {date}: {result.message}")
                        # Fallback sur poids égaux
                        weights.loc[date, selected_tickers] = 1.0 / n_assets
                
            return weights
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction du portefeuille à Sharpe maximal: {e}")
            return pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
    
    @staticmethod
    def risk_parity(selected_assets, returns):
        """
        Construit un portefeuille à parité de risque des actifs sélectionnés.
        
        Args:
            selected_assets (pandas.DataFrame): DataFrame booléen indiquant les actifs sélectionnés
            returns (pandas.DataFrame): Rendements historiques des actifs
            
        Returns:
            pandas.DataFrame: Poids du portefeuille pour chaque actif et chaque date
        """
        try:
            # Initialiser les poids à zéro
            weights = pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
            
            # Pour chaque date, optimiser les poids pour égaliser les contributions au risque
            for date in weights.index:
                # Filtrer les actifs sélectionnés
                mask = selected_assets.loc[date]
                selected_tickers = mask[mask].index.tolist()
                
                if len(selected_tickers) > 0:
                    # Calculer la matrice de covariance
                    lookback_end = date
                    lookback_start_idx = max(0, returns.index.get_loc(lookback_end) - 252)
                    lookback_start = returns.index[lookback_start_idx]
                    
                    historical_returns = returns.loc[lookback_start:lookback_end, selected_tickers]
                    cov_matrix = historical_returns.cov()
                    
                    # Définir la fonction objectif pour la parité de risque
                    def risk_parity_objective(w):
                        w = np.maximum(w, 0)  # Assurer des poids positifs
                        w = w / np.sum(w)     # Normaliser pour sommer à 1
                        
                        # Calculer les contributions au risque
                        portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                        marginal_contrib = np.dot(cov_matrix, w) / portfolio_vol
                        risk_contrib = w * marginal_contrib
                        
                        # Objectif: minimiser la somme des écarts carrés des contributions au risque
                        target_risk = portfolio_vol / len(w)
                        return np.sum((risk_contrib - target_risk) ** 2)
                    
                    # Contraintes: les poids doivent sommer à 1 et être positifs
                    n_assets = len(selected_tickers)
                    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                    bounds = tuple((0, 1) for _ in range(n_assets))
                    
                    # Poids initiaux égaux
                    initial_weights = np.ones(n_assets) / n_assets
                    
                    # Optimiser
                    result = minimize(
                        risk_parity_objective,
                        initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                    
                    if result.success:
                        optimized_weights = result.x
                        weights.loc[date, selected_tickers] = optimized_weights
                    else:
                        logger.warning(f"Échec de l'optimisation à la date {date}: {result.message}")
                        # Fallback sur poids égaux
                        weights.loc[date, selected_tickers] = 1.0 / n_assets
                
            return weights
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction du portefeuille à parité de risque: {e}")
            return pd.DataFrame(0, index=selected_assets.index, columns=selected_assets.columns)
    
    @staticmethod
    def apply_constraints(weights, min_weight=0.01, max_weight=0.2):
        """
        Applique des contraintes de poids minimum et maximum au portefeuille.
        
        Args:
            weights (pandas.DataFrame): Poids du portefeuille
            min_weight (float): Poids minimum pour un actif (si non nul)
            max_weight (float): Poids maximum pour un actif
            
        Returns:
            pandas.DataFrame: Poids du portefeuille contraints
        """
        try:
            constrained_weights = weights.copy()
            
            for date in constrained_weights.index:
                # Récupérer les poids non nuls
                non_zero_weights = constrained_weights.loc[date][constrained_weights.loc[date] > 0]
                
                if len(non_zero_weights) > 0:
                    # Appliquer le poids minimum
                    too_small = (non_zero_weights < min_weight) & (non_zero_weights > 0)
                    
                    if too_small.any():
                        # Éliminer les positions trop petites
                        constrained_weights.loc[date, too_small.index] = 0
                        
                        # Recalculer les positions restantes
                        remaining = constrained_weights.loc[date] > 0
                        
                        if remaining.sum() > 0:
                            # Normaliser pour que la somme soit à nouveau 1
                            constrained_weights.loc[date, remaining] = constrained_weights.loc[date, remaining] / constrained_weights.loc[date, remaining].sum()
                    
                    # Appliquer le poids maximum
                    too_large = constrained_weights.loc[date] > max_weight
                    
                    if too_large.any():
                        excess = constrained_weights.loc[date, too_large] - max_weight
                        constrained_weights.loc[date, too_large] = max_weight
                        
                        # Redistribuer l'excès aux autres positions
                        smaller_positions = (constrained_weights.loc[date] < max_weight) & (constrained_weights.loc[date] > 0)
                        
                        if smaller_positions.sum() > 0:
                            to_distribute = excess.sum()
                            weights_to_add = to_distribute * constrained_weights.loc[date, smaller_positions] / constrained_weights.loc[date, smaller_positions].sum()
                            constrained_weights.loc[date, smaller_positions] += weights_to_add
            
            return constrained_weights
            
        except Exception as e:
            logger.error(f"Erreur lors de l'application des contraintes: {e}")
            return weights
