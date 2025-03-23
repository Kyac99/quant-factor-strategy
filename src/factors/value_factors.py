"""
Module implémentant les facteurs de value pour la stratégie d'investissement.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ValueFactors:
    """
    Classe pour calculer les facteurs de value à partir des données fondamentales.
    """
    
    @staticmethod
    def calculate_pe_ratio(price_data, earnings_data):
        """
        Calcule le ratio Price-to-Earnings (P/E) pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            earnings_data (dict): Dictionnaire avec les données de résultats par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les ratios P/E
        """
        pe_ratios = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        
        for ticker in price_data.columns:
            try:
                if ticker in earnings_data:
                    # Récupérer les derniers résultats annuels par action
                    eps = earnings_data[ticker].get('eps', None)
                    
                    if eps and eps != 0:
                        # Calculer P/E pour chaque période
                        pe_ratios[ticker] = price_data[ticker] / eps
                    else:
                        logger.warning(f"Pas de donnée EPS valide pour {ticker}")
                        pe_ratios[ticker] = np.nan
                else:
                    logger.warning(f"Pas de données de résultats pour {ticker}")
                    pe_ratios[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du P/E pour {ticker}: {e}")
                pe_ratios[ticker] = np.nan
                
        return pe_ratios
        
    @staticmethod
    def calculate_pb_ratio(price_data, book_value_data):
        """
        Calcule le ratio Price-to-Book (P/B) pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            book_value_data (dict): Dictionnaire avec les données de valeur comptable par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les ratios P/B
        """
        pb_ratios = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        
        for ticker in price_data.columns:
            try:
                if ticker in book_value_data:
                    # Récupérer la dernière valeur comptable par action
                    bvps = book_value_data[ticker].get('bookValue', None)
                    
                    if bvps and bvps != 0:
                        # Calculer P/B pour chaque période
                        pb_ratios[ticker] = price_data[ticker] / bvps
                    else:
                        logger.warning(f"Pas de donnée de valeur comptable valide pour {ticker}")
                        pb_ratios[ticker] = np.nan
                else:
                    logger.warning(f"Pas de données de valeur comptable pour {ticker}")
                    pb_ratios[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du P/B pour {ticker}: {e}")
                pb_ratios[ticker] = np.nan
                
        return pb_ratios
        
    @staticmethod
    def calculate_ev_ebitda(price_data, fundamentals):
        """
        Calcule le ratio Enterprise Value to EBITDA pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les ratios EV/EBITDA
        """
        ev_ebitda = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        
        for ticker in price_data.columns:
            try:
                if ticker in fundamentals:
                    # Récupérer les dernières infos fondamentales
                    info = fundamentals[ticker].get('info', {})
                    
                    # Récupérer Enterprise Value et EBITDA
                    ev = info.get('enterpriseValue', None)
                    ebitda = info.get('ebitda', None)
                    
                    if ev and ebitda and ebitda != 0:
                        # Le ratio est constant pour toutes les périodes car nous n'avons que les dernières données
                        ev_ebitda[ticker] = pd.Series([ev / ebitda] * len(price_data.index), index=price_data.index)
                    else:
                        logger.warning(f"Données EV ou EBITDA manquantes ou invalides pour {ticker}")
                        ev_ebitda[ticker] = np.nan
                else:
                    logger.warning(f"Pas de données fondamentales pour {ticker}")
                    ev_ebitda[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du EV/EBITDA pour {ticker}: {e}")
                ev_ebitda[ticker] = np.nan
                
        return ev_ebitda
        
    @staticmethod
    def calculate_dividend_yield(price_data, dividend_data):
        """
        Calcule le rendement du dividende pour chaque ticker.
        
        Args:
            price_data (pandas.DataFrame): DataFrame contenant les prix de clôture
            dividend_data (dict): Dictionnaire avec les données de dividendes par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les rendements de dividendes
        """
        div_yield = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        
        for ticker in price_data.columns:
            try:
                if ticker in dividend_data:
                    # Récupérer le dividende annuel
                    div_annual = dividend_data[ticker].get('dividendRate', 0)
                    
                    if div_annual > 0:
                        # Calculer le rendement pour chaque période
                        div_yield[ticker] = (div_annual / price_data[ticker]) * 100
                    else:
                        logger.info(f"Pas de dividende pour {ticker}")
                        div_yield[ticker] = 0
                else:
                    logger.warning(f"Pas de données de dividendes pour {ticker}")
                    div_yield[ticker] = 0
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du rendement de dividende pour {ticker}: {e}")
                div_yield[ticker] = 0
                
        return div_yield
        
    @staticmethod
    def combine_value_factors(pe_ratio, pb_ratio, ev_ebitda, div_yield=None, weights=None):
        """
        Combine les différents facteurs de value en un score composite.
        
        Args:
            pe_ratio (pandas.DataFrame): Ratios P/E
            pb_ratio (pandas.DataFrame): Ratios P/B
            ev_ebitda (pandas.DataFrame): Ratios EV/EBITDA
            div_yield (pandas.DataFrame, optional): Rendements de dividendes
            weights (dict, optional): Poids à appliquer à chaque facteur
            
        Returns:
            pandas.DataFrame: Score de value composite
        """
        # Poids par défaut
        if weights is None:
            weights = {'pe': 0.3, 'pb': 0.3, 'ev_ebitda': 0.3, 'div_yield': 0.1}
            
        # Normaliser les facteurs (plus petit est meilleur pour PE, PB, EV/EBITDA)
        factors = {}
        
        # Pour P/E, P/B et EV/EBITDA, plus petit est meilleur
        for name, data, reverse in [
            ('pe', pe_ratio, True),
            ('pb', pb_ratio, True),
            ('ev_ebitda', ev_ebitda, True),
            ('div_yield', div_yield, False)  # pour div_yield, plus grand est meilleur
        ]:
            if data is not None:
                # Remplacer les infinis et NaN par la médiane
                median_values = data.replace([np.inf, -np.inf], np.nan).median()
                cleaned_data = data.copy()
                
                for col in cleaned_data.columns:
                    mask = cleaned_data[col].isin([np.inf, -np.inf]) | cleaned_data[col].isna()
                    cleaned_data.loc[mask, col] = median_values[col]
                
                # Z-score normalization
                normalized = (cleaned_data - cleaned_data.mean()) / cleaned_data.std()
                
                # Inverser si nécessaire (pour que plus petit soit meilleur)
                if reverse:
                    normalized = -normalized
                
                factors[name] = normalized
        
        # Combiner avec les poids
        composite_score = pd.DataFrame(0, index=pe_ratio.index, columns=pe_ratio.columns)
        
        for name, factor in factors.items():
            if name in weights and weights[name] > 0:
                composite_score += factor * weights[name]
                
        return composite_score
