"""
Module principal pour combiner les différents facteurs en un modèle d'évaluation complet.
"""
import logging
import pandas as pd
import numpy as np

from src.factors.value_factors import ValueFactors
from src.factors.momentum_factors import MomentumFactors
from src.factors.quality_factors import QualityFactors
from src.factors.low_volatility_factors import LowVolatilityFactors

logger = logging.getLogger(__name__)

class FactorModel:
    """
    Classe principale pour combiner différents facteurs et construire un modèle multifactoriel.
    """
    
    def __init__(self, factor_weights=None):
        """
        Initialise le modèle factoriel.
        
        Args:
            factor_weights (dict, optional): Poids à appliquer à chaque type de facteur
        """
        # Poids par défaut si non spécifiés
        self.factor_weights = factor_weights or {
            'value': 0.25,
            'momentum': 0.25,
            'quality': 0.25,
            'low_volatility': 0.25
        }
        
        # Vérifier que les poids somment à 1
        total_weight = sum(self.factor_weights.values())
        if abs(total_weight - 1.0) > 1e-10:
            logger.warning(f"Les poids des facteurs ne somment pas à 1 (total = {total_weight}). Normalisation appliquée.")
            for key in self.factor_weights:
                self.factor_weights[key] /= total_weight
    
    def calculate_value_score(self, price_data, fundamentals, value_weights=None):
        """
        Calcule le score de value composite.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            value_weights (dict, optional): Poids spécifiques pour les facteurs de value
            
        Returns:
            pandas.DataFrame: Score de value composite
        """
        try:
            # Extraire les données nécessaires des fondamentaux
            earnings_data = {}
            book_value_data = {}
            dividend_data = {}
            
            for ticker, data in fundamentals.items():
                if 'info' in data:
                    # Données de résultats
                    eps = data['info'].get('trailingEPS', None)
                    earnings_data[ticker] = {'eps': eps}
                    
                    # Valeur comptable
                    book_value = data['info'].get('bookValue', None)
                    book_value_data[ticker] = {'bookValue': book_value}
                    
                    # Dividendes
                    div_rate = data['info'].get('dividendRate', 0)
                    dividend_data[ticker] = {'dividendRate': div_rate}
            
            # Calculer les facteurs de value
            pe_ratio = ValueFactors.calculate_pe_ratio(price_data, earnings_data)
            pb_ratio = ValueFactors.calculate_pb_ratio(price_data, book_value_data)
            ev_ebitda = ValueFactors.calculate_ev_ebitda(price_data, fundamentals)
            div_yield = ValueFactors.calculate_dividend_yield(price_data, dividend_data)
            
            # Combiner en un score de value
            value_score = ValueFactors.combine_value_factors(
                pe_ratio, pb_ratio, ev_ebitda, div_yield, weights=value_weights
            )
            
            return value_score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de value: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    def calculate_momentum_score(self, price_data, momentum_weights=None):
        """
        Calcule le score de momentum composite.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            momentum_weights (dict, optional): Poids spécifiques pour les facteurs de momentum
            
        Returns:
            pandas.DataFrame: Score de momentum composite
        """
        try:
            # Calculer les facteurs de momentum
            momentum_12_1 = MomentumFactors.calculate_momentum_12_1(price_data)
            
            # Calculer les accélérations de prix
            acceleration = MomentumFactors.calculate_price_acceleration(price_data)
            
            # Calculer le RSI
            rsi = MomentumFactors.calculate_relative_strength_index(price_data)
            
            # Combiner en un score de momentum
            momentum_score = MomentumFactors.combine_momentum_factors(
                momentum_12_1, acceleration, rsi, weights=momentum_weights
            )
            
            return momentum_score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de momentum: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    def calculate_quality_score(self, fundamentals, quality_weights=None):
        """
        Calcule le score de qualité composite.
        
        Args:
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            quality_weights (dict, optional): Poids spécifiques pour les facteurs de qualité
            
        Returns:
            pandas.DataFrame: Score de qualité composite
        """
        try:
            # Calculer les facteurs de qualité
            roe = QualityFactors.calculate_roe(fundamentals)
            roa = QualityFactors.calculate_roa(fundamentals)
            profit_margin = QualityFactors.calculate_profit_margin(fundamentals)
            debt_to_equity = QualityFactors.calculate_debt_to_equity(fundamentals)
            interest_coverage = QualityFactors.calculate_interest_coverage(fundamentals)
            
            # Combiner en un score de qualité
            quality_score = QualityFactors.combine_quality_factors(
                roe, roa, profit_margin, debt_to_equity, interest_coverage, weights=quality_weights
            )
            
            # Dupliquer le score pour toutes les dates (car les fondamentaux sont statiques)
            # Cela suppose que price_data a été utilisé ailleurs et a un index de dates
            # À ajuster selon le contexte d'utilisation
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de qualité: {e}")
            return pd.DataFrame()
    
    def calculate_low_volatility_score(self, price_data, market_data, low_vol_weights=None):
        """
        Calcule le score de faible volatilité composite.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            market_data (pandas.Series): Série contenant les prix de clôture de l'indice de marché
            low_vol_weights (dict, optional): Poids spécifiques pour les facteurs de faible volatilité
            
        Returns:
            pandas.DataFrame: Score de faible volatilité composite
        """
        try:
            # Calculer les facteurs de faible volatilité
            volatility = LowVolatilityFactors.calculate_volatility(price_data)
            beta = LowVolatilityFactors.calculate_beta(price_data, market_data)
            downside_volatility = LowVolatilityFactors.calculate_downside_volatility(price_data)
            max_drawdown = LowVolatilityFactors.calculate_max_drawdown(price_data)
            sortino_ratio = LowVolatilityFactors.calculate_sortino_ratio(price_data)
            
            # Combiner en un score de faible volatilité
            low_vol_score = LowVolatilityFactors.combine_low_volatility_factors(
                volatility, beta, downside_volatility, max_drawdown, sortino_ratio, weights=low_vol_weights
            )
            
            return low_vol_score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de faible volatilité: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    def calculate_combined_score(self, price_data, fundamentals, market_data,
                                value_weights=None, momentum_weights=None,
                                quality_weights=None, low_vol_weights=None):
        """
        Calcule un score combiné à partir de tous les facteurs.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            market_data (pandas.Series): Série contenant les prix de clôture de l'indice de marché
            value_weights (dict, optional): Poids pour les facteurs de value
            momentum_weights (dict, optional): Poids pour les facteurs de momentum
            quality_weights (dict, optional): Poids pour les facteurs de qualité
            low_vol_weights (dict, optional): Poids pour les facteurs de faible volatilité
            
        Returns:
            pandas.DataFrame: Score combiné pour tous les facteurs
        """
        try:
            # Calculer les scores pour chaque type de facteur
            value_score = self.calculate_value_score(price_data, fundamentals, value_weights)
            momentum_score = self.calculate_momentum_score(price_data, momentum_weights)
            quality_score = self.calculate_quality_score(fundamentals, quality_weights)
            low_vol_score = self.calculate_low_volatility_score(price_data, market_data, low_vol_weights)
            
            # Dupliquer le score de qualité pour toutes les dates si nécessaire
            if not quality_score.empty and len(quality_score) == 1:
                quality_score = pd.DataFrame(
                    np.tile(quality_score.values, (len(price_data), 1)),
                    index=price_data.index,
                    columns=quality_score.columns
                )
            
            # Initialiser le score combiné
            combined_score = pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
            
            # Ajouter chaque score avec son poids
            if not value_score.empty and 'value' in self.factor_weights:
                combined_score += value_score * self.factor_weights['value']
                
            if not momentum_score.empty and 'momentum' in self.factor_weights:
                combined_score += momentum_score * self.factor_weights['momentum']
                
            if not quality_score.empty and 'quality' in self.factor_weights:
                combined_score += quality_score * self.factor_weights['quality']
                
            if not low_vol_score.empty and 'low_volatility' in self.factor_weights:
                combined_score += low_vol_score * self.factor_weights['low_volatility']
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score combiné: {e}")
            return pd.DataFrame(np.nan, index=price_data.index, columns=price_data.columns)
    
    def rank_assets(self, combined_score, ascending=False):
        """
        Classe les actifs en fonction de leur score combiné.
        
        Args:
            combined_score (pandas.DataFrame): Score combiné pour tous les facteurs
            ascending (bool): Si True, les scores plus faibles sont mieux classés
            
        Returns:
            pandas.DataFrame: Classements des actifs
        """
        try:
            # Classer les actifs à chaque date
            ranks = combined_score.rank(axis=1, ascending=ascending)
            
            return ranks
            
        except Exception as e:
            logger.error(f"Erreur lors du classement des actifs: {e}")
            return pd.DataFrame(np.nan, index=combined_score.index, columns=combined_score.columns)
    
    def select_top_assets(self, ranks, top_n=10):
        """
        Sélectionne les N meilleurs actifs en fonction de leur classement.
        
        Args:
            ranks (pandas.DataFrame): Classements des actifs
            top_n (int): Nombre d'actifs à sélectionner
            
        Returns:
            pandas.DataFrame: DataFrame booléen indiquant les actifs sélectionnés
        """
        try:
            # Sélectionner les top_n actifs à chaque date
            selected = ranks <= top_n
            
            return selected
            
        except Exception as e:
            logger.error(f"Erreur lors de la sélection des meilleurs actifs: {e}")
            return pd.DataFrame(False, index=ranks.index, columns=ranks.columns)
