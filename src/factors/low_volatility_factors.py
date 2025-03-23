"""
Module implémentant les facteurs de faible volatilité pour la stratégie d'investissement.
"""
import logging
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

logger = logging.getLogger(__name__)

class LowVolatilityFactors:
    """
    Classe pour calculer les facteurs de faible volatilité à partir des données de prix.
    """
    
    @staticmethod
    def calculate_volatility(price_data, window=252):
        """
        Calcule la volatilité historique pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            window (int): Fenêtre pour le calcul de la volatilité (par défaut 252 jours, soit ~1 an)
            
        Returns:
            pandas.DataFrame: DataFrame contenant les volatilités annualisées
        """
        try:
            # Calculer les rendements quotidiens
            returns = price_data.pct_change().dropna()
            
            # Calculer la volatilité sur la fenêtre spécifiée et annualiser
            # Volatilité annualisée = volatilité quotidienne * sqrt(252)
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la volatilité: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    @staticmethod
    def calculate_beta(price_data, market_data, window=252):
        """
        Calcule le beta par rapport à l'indice de marché pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture des actifs
            market_data (pandas.Series): Série contenant les prix de clôture de l'indice de marché
            window (int): Fenêtre pour le calcul du beta (par défaut 252 jours, soit ~1 an)
            
        Returns:
            pandas.DataFrame: DataFrame contenant les valeurs de beta
        """
        try:
            # Calculer les rendements quotidiens
            asset_returns = price_data.pct_change().dropna()
            market_returns = market_data.pct_change().dropna()
            
            # S'assurer que les deux DataFrames ont le même index
            common_index = asset_returns.index.intersection(market_returns.index)
            asset_returns = asset_returns.loc[common_index]
            market_returns = market_returns.loc[common_index]
            
            # Initialiser le DataFrame pour les betas
            beta = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns)
            
            # Calculer le beta pour chaque ticker
            for ticker in asset_returns.columns:
                # Utiliser une régression glissante pour calculer le beta sur la fenêtre spécifiée
                for i in range(window, len(asset_returns) + 1):
                    start_idx = i - window
                    end_idx = i
                    
                    if start_idx >= 0 and end_idx <= len(asset_returns):
                        X = market_returns.iloc[start_idx:end_idx].values
                        y = asset_returns[ticker].iloc[start_idx:end_idx].values
                        
                        # Ajouter une constante pour la régression
                        X = add_constant(X)
                        
                        try:
                            # Effectuer la régression
                            model = OLS(y, X).fit()
                            
                            # Le beta est le coefficient de la variable explicative (marché)
                            current_beta = model.params[1]
                            
                            # Stocker le beta calculé
                            beta.loc[asset_returns.index[end_idx-1], ticker] = current_beta
                        except Exception as e:
                            logger.warning(f"Erreur lors de la régression pour {ticker} à l'indice {end_idx}: {e}")
                            beta.loc[asset_returns.index[end_idx-1], ticker] = np.nan
            
            # Propager les valeurs de beta pour les périodes précédant la première fenêtre
            beta.fillna(method='bfill', inplace=True)
                
            return beta
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du beta: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    @staticmethod
    def calculate_downside_volatility(price_data, market_data=None, threshold=0, window=252):
        """
        Calcule la volatilité à la baisse pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            market_data (pandas.Series, optional): Série contenant les prix de clôture de l'indice de marché
            threshold (float): Seuil pour les rendements considérés comme "à la baisse" (par défaut 0)
            window (int): Fenêtre pour le calcul (par défaut 252 jours, soit ~1 an)
            
        Returns:
            pandas.DataFrame: DataFrame contenant les volatilités à la baisse
        """
        try:
            # Calculer les rendements quotidiens
            returns = price_data.pct_change().dropna()
            
            # Si market_data est fourni, calculer par rapport au marché
            if market_data is not None:
                market_returns = market_data.pct_change().dropna()
                
                # S'assurer que les deux DataFrames ont le même index
                common_index = returns.index.intersection(market_returns.index)
                returns = returns.loc[common_index]
                market_returns = market_returns.loc[common_index]
                
                # Calculer les rendements excédentaires par rapport au marché
                excess_returns = returns.sub(market_returns, axis=0)
                
                # Ne considérer que les rendements inférieurs au seuil
                downside_returns = excess_returns.where(excess_returns < threshold, 0)
            else:
                # Ne considérer que les rendements inférieurs au seuil
                downside_returns = returns.where(returns < threshold, 0)
            
            # Calculer la volatilité à la baisse (racine carrée de la moyenne des carrés des rendements négatifs)
            downside_volatility = downside_returns.rolling(window=window).apply(
                lambda x: np.sqrt((x ** 2).mean()) * np.sqrt(252),
                raw=True
            )
            
            return downside_volatility
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la volatilité à la baisse: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    @staticmethod
    def calculate_max_drawdown(price_data, window=252):
        """
        Calcule le drawdown maximal pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            window (int): Fenêtre pour le calcul (par défaut 252 jours, soit ~1 an)
            
        Returns:
            pandas.DataFrame: DataFrame contenant les drawdowns maximaux
        """
        try:
            # Initialiser le DataFrame de résultat
            max_drawdown = pd.DataFrame(index=price_data.index, columns=price_data.columns)
            
            for ticker in price_data.columns:
                # Calculer les drawdowns sur une fenêtre glissante
                for i in range(window, len(price_data) + 1):
                    start_idx = i - window
                    end_idx = i
                    
                    if start_idx >= 0 and end_idx <= len(price_data):
                        # Récupérer les prix sur la fenêtre
                        window_prices = price_data[ticker].iloc[start_idx:end_idx]
                        
                        # Calculer le drawdown maximal
                        cum_max = window_prices.cummax()
                        drawdown = (window_prices - cum_max) / cum_max
                        current_max_drawdown = drawdown.min()
                        
                        # Stocker le drawdown maximal
                        max_drawdown.loc[price_data.index[end_idx-1], ticker] = current_max_drawdown
            
            # Propager les valeurs pour les périodes précédant la première fenêtre
            max_drawdown.fillna(method='bfill', inplace=True)
                
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du drawdown maximal: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    @staticmethod
    def calculate_sortino_ratio(price_data, risk_free_rate=0.0, threshold=0.0, window=252):
        """
        Calcule le ratio de Sortino pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            risk_free_rate (float): Taux sans risque annualisé
            threshold (float): Seuil pour les rendements considérés comme "à la baisse"
            window (int): Fenêtre pour le calcul (par défaut 252 jours, soit ~1 an)
            
        Returns:
            pandas.DataFrame: DataFrame contenant les ratios de Sortino
        """
        try:
            # Calculer les rendements quotidiens
            returns = price_data.pct_change().dropna()
            
            # Convertir le taux sans risque annualisé en taux quotidien
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            
            # Initialiser le DataFrame de résultat
            sortino_ratio = pd.DataFrame(index=returns.index, columns=returns.columns)
            
            for ticker in returns.columns:
                # Calculer le ratio de Sortino sur une fenêtre glissante
                for i in range(window, len(returns) + 1):
                    start_idx = i - window
                    end_idx = i
                    
                    if start_idx >= 0 and end_idx <= len(returns):
                        # Récupérer les rendements sur la fenêtre
                        window_returns = returns[ticker].iloc[start_idx:end_idx]
                        
                        # Calculer le rendement moyen excédentaire
                        excess_return = window_returns.mean() - daily_risk_free
                        
                        # Calculer la volatilité à la baisse
                        downside_returns = window_returns.where(window_returns < threshold, 0)
                        downside_vol = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
                        
                        # Calculer le ratio de Sortino
                        if downside_vol != 0:
                            current_sortino = (excess_return * 252) / downside_vol
                        else:
                            current_sortino = np.nan
                        
                        # Stocker le ratio de Sortino
                        sortino_ratio.loc[returns.index[end_idx-1], ticker] = current_sortino
            
            # Propager les valeurs pour les périodes précédant la première fenêtre
            sortino_ratio.fillna(method='bfill', inplace=True)
                
            return sortino_ratio
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du ratio de Sortino: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    @staticmethod
    def combine_low_volatility_factors(volatility, beta, downside_volatility=None, max_drawdown=None, sortino_ratio=None, weights=None):
        """
        Combine les différents facteurs de faible volatilité en un score composite.
        
        Args:
            volatility (pandas.DataFrame): Volatilité historique
            beta (pandas.DataFrame): Beta par rapport au marché
            downside_volatility (pandas.DataFrame, optional): Volatilité à la baisse
            max_drawdown (pandas.DataFrame, optional): Drawdown maximal
            sortino_ratio (pandas.DataFrame, optional): Ratio de Sortino
            weights (dict, optional): Poids à appliquer à chaque facteur
            
        Returns:
            pandas.DataFrame: Score de faible volatilité composite
        """
        # Poids par défaut
        if weights is None:
            weights = {
                'volatility': 0.3, 
                'beta': 0.3, 
                'downside_volatility': 0.2, 
                'max_drawdown': 0.1, 
                'sortino_ratio': 0.1
            }
            
        # Normaliser les facteurs
        factors = {}
        
        # Pour volatility, beta, downside_volatility et max_drawdown, plus petit est meilleur
        # Pour sortino_ratio, plus grand est meilleur
        for name, data, reverse in [
            ('volatility', volatility, True),
            ('beta', beta, True),
            ('downside_volatility', downside_volatility, True),
            ('max_drawdown', max_drawdown, True),
            ('sortino_ratio', sortino_ratio, False)
        ]:
            if data is not None:
                # Remplacer les infinis et NaN
                cleaned_data = data.replace([np.inf, -np.inf], np.nan)
                
                # Remplir les NaN avec la médiane
                for col in cleaned_data.columns:
                    median_val = cleaned_data[col].median()
                    cleaned_data[col].fillna(median_val, inplace=True)
                
                # Z-score normalization
                normalized = (cleaned_data - cleaned_data.mean()) / cleaned_data.std()
                
                # Inverser si nécessaire
                if reverse:
                    normalized = -normalized
                
                factors[name] = normalized
        
        # Combiner avec les poids
        composite_score = pd.DataFrame(0, index=volatility.index, columns=volatility.columns)
        
        for name, factor in factors.items():
            if name in weights and weights[name] > 0:
                composite_score += factor * weights[name]
                
        return composite_score
