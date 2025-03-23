"""
Module implémentant les facteurs de momentum pour la stratégie d'investissement.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MomentumFactors:
    """
    Classe pour calculer les facteurs de momentum à partir des données de prix.
    """
    
    @staticmethod
    def calculate_returns(price_data, periods=[1, 3, 6, 12]):
        """
        Calcule les rendements sur différentes périodes pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            periods (list): Liste des périodes en mois pour calculer les rendements
            
        Returns:
            dict: Dictionnaire de DataFrames, un pour chaque période
        """
        returns = {}
        
        for period in periods:
            try:
                # Convertir les mois en périodes (en supposant des données quotidiennes)
                offset = period * 21  # ~21 jours de trading par mois
                
                # Calculer les rendements
                period_returns = price_data.pct_change(offset)
                
                returns[period] = period_returns
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul des rendements sur {period} mois: {e}")
                
        return returns
        
    @staticmethod
    def calculate_momentum_12_1(price_data):
        """
        Calcule le momentum 12-1 (rendement sur 12 mois excluant le dernier mois).
        C'est une mesure classique du momentum.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            
        Returns:
            pandas.DataFrame: DataFrame contenant le momentum 12-1
        """
        try:
            # Rendement sur 12 mois
            returns_12m = price_data.pct_change(252)  # ~252 jours de trading par an
            
            # Rendement sur 1 mois
            returns_1m = price_data.pct_change(21)    # ~21 jours de trading par mois
            
            # Momentum 12-1 : rendement sur 12 mois ajusté pour exclure le dernier mois
            momentum_12_1 = ((1 + returns_12m) / (1 + returns_1m)) - 1
            
            return momentum_12_1
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du momentum 12-1: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    @staticmethod
    def calculate_momentum_score(returns_data, weights=None):
        """
        Calcule un score de momentum composite à partir des rendements sur différentes périodes.
        
        Args:
            returns_data (dict): Dictionnaire de DataFrames de rendements par période
            weights (dict, optional): Poids à appliquer à chaque période
            
        Returns:
            pandas.DataFrame: Score de momentum composite
        """
        if not returns_data:
            return pd.DataFrame()
            
        # Poids par défaut si non spécifiés
        if weights is None:
            # En général, on donne plus de poids aux périodes récentes
            total_periods = len(returns_data)
            weights = {}
            for i, period in enumerate(sorted(returns_data.keys())):
                # Formule simple qui donne plus de poids aux périodes plus courtes
                weights[period] = (total_periods - i) / sum(range(1, total_periods + 1))
                
        # Récupérer un DataFrame comme référence pour l'index et les colonnes
        first_df = next(iter(returns_data.values()))
        
        # Initialiser le score composite
        momentum_score = pd.DataFrame(0, index=first_df.index, columns=first_df.columns)
        
        # Normaliser chaque période et l'ajouter au score composite
        for period, returns in returns_data.items():
            if period in weights:
                # Remplacer les valeurs non finies
                cleaned_returns = returns.replace([np.inf, -np.inf], np.nan)
                
                # Remplir les NaN avec des zéros pour éviter les problèmes de calcul
                cleaned_returns.fillna(0, inplace=True)
                
                # Normaliser (z-score) pour rendre les périodes comparables
                z_scored = (cleaned_returns - cleaned_returns.mean()) / cleaned_returns.std()
                
                # Ajouter au score composite avec le poids approprié
                momentum_score += z_scored * weights[period]
                
        return momentum_score
        
    @staticmethod
    def calculate_price_acceleration(price_data, short_period=1, medium_period=3, long_period=6):
        """
        Calcule l'accélération du prix, qui est la différence entre le momentum récent et le momentum à plus long terme.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            short_period (int): Période courte en mois
            medium_period (int): Période moyenne en mois
            long_period (int): Période longue en mois
            
        Returns:
            pandas.DataFrame: DataFrame contenant l'accélération du prix
        """
        try:
            # Calculer les rendements sur différentes périodes
            short_returns = price_data.pct_change(short_period * 21)
            medium_returns = price_data.pct_change(medium_period * 21)
            long_returns = price_data.pct_change(long_period * 21)
            
            # Calculer l'accélération du prix
            # L'idée est que si le momentum récent est plus fort que le momentum à long terme,
            # cela peut indiquer une accélération de la tendance
            acceleration = (short_returns - medium_returns) + (medium_returns - long_returns)
            
            return acceleration
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'accélération du prix: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    @staticmethod
    def calculate_relative_strength_index(price_data, window=14):
        """
        Calcule l'indice de force relative (RSI) pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            window (int): Fenêtre pour le calcul du RSI (typiquement 14 jours)
            
        Returns:
            pandas.DataFrame: DataFrame contenant les valeurs RSI
        """
        # Initialiser le DataFrame de résultat
        rsi = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        
        for ticker in price_data.columns:
            try:
                # Calculer les variations quotidiennes
                delta = price_data[ticker].diff()
                
                # Séparer les hausses et les baisses
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                # Calculer la moyenne mobile des gains et des pertes
                avg_gain = gain.rolling(window=window, min_periods=1).mean()
                avg_loss = loss.rolling(window=window, min_periods=1).mean()
                
                # Calculer le ratio
                rs = avg_gain / avg_loss
                
                # Calculer le RSI
                rsi[ticker] = 100 - (100 / (1 + rs))
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul du RSI pour {ticker}: {e}")
                rsi[ticker] = np.nan
                
        return rsi
        
    @staticmethod
    def combine_momentum_factors(momentum_12_1, acceleration, rsi, weights=None):
        """
        Combine les différents facteurs de momentum en un score composite.
        
        Args:
            momentum_12_1 (pandas.DataFrame): Momentum 12-1
            acceleration (pandas.DataFrame): Accélération du prix
            rsi (pandas.DataFrame): Indice de force relative
            weights (dict, optional): Poids à appliquer à chaque facteur
            
        Returns:
            pandas.DataFrame: Score de momentum composite
        """
        # Poids par défaut
        if weights is None:
            weights = {'momentum_12_1': 0.5, 'acceleration': 0.3, 'rsi': 0.2}
            
        # Normaliser les facteurs
        factors = {}
        
        for name, data in [
            ('momentum_12_1', momentum_12_1),
            ('acceleration', acceleration),
            ('rsi', rsi)
        ]:
            if data is not None:
                # Remplacer les infinis et NaN
                cleaned_data = data.replace([np.inf, -np.inf], np.nan)
                cleaned_data.fillna(method='ffill', inplace=True)
                cleaned_data.fillna(method='bfill', inplace=True)
                
                # Z-score normalization
                normalized = (cleaned_data - cleaned_data.mean()) / cleaned_data.std()
                
                factors[name] = normalized
        
        # Combiner avec les poids
        composite_score = pd.DataFrame(0, index=momentum_12_1.index, columns=momentum_12_1.columns)
        
        for name, factor in factors.items():
            if name in weights and weights[name] > 0:
                composite_score += factor * weights[name]
                
        return composite_score
