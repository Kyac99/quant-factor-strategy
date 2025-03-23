"""
Module implémentant les facteurs de qualité pour la stratégie d'investissement.
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class QualityFactors:
    """
    Classe pour calculer les facteurs de qualité à partir des données fondamentales.
    """
    
    @staticmethod
    def calculate_roe(fundamentals):
        """
        Calcule le Return on Equity (ROE) pour chaque ticker.
        
        Args:
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les valeurs ROE
        """
        roe_data = {}
        
        for ticker, data in fundamentals.items():
            try:
                if 'info' in data:
                    # Récupérer le ROE directement si disponible
                    roe = data['info'].get('returnOnEquity', None)
                    
                    if roe is not None:
                        roe_data[ticker] = roe
                    else:
                        # Calculer le ROE à partir des états financiers si non disponible directement
                        if 'income_stmt' in data and 'balance_sheet' in data:
                            # Récupérer le bénéfice net
                            net_income = data['income_stmt'].loc['Net Income'].iloc[0]
                            
                            # Récupérer les capitaux propres
                            equity = data['balance_sheet'].loc['Total Stockholder Equity'].iloc[0]
                            
                            if equity != 0:
                                roe = net_income / equity
                                roe_data[ticker] = roe
                            else:
                                logger.warning(f"Capitaux propres nuls pour {ticker}")
                                roe_data[ticker] = np.nan
                        else:
                            logger.warning(f"Données insuffisantes pour calculer le ROE pour {ticker}")
                            roe_data[ticker] = np.nan
                else:
                    logger.warning(f"Informations fondamentales manquantes pour {ticker}")
                    roe_data[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du ROE pour {ticker}: {e}")
                roe_data[ticker] = np.nan
                
        return pd.DataFrame([roe_data])
        
    @staticmethod
    def calculate_roa(fundamentals):
        """
        Calcule le Return on Assets (ROA) pour chaque ticker.
        
        Args:
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les valeurs ROA
        """
        roa_data = {}
        
        for ticker, data in fundamentals.items():
            try:
                if 'info' in data:
                    # Récupérer le ROA directement si disponible
                    roa = data['info'].get('returnOnAssets', None)
                    
                    if roa is not None:
                        roa_data[ticker] = roa
                    else:
                        # Calculer le ROA à partir des états financiers si non disponible directement
                        if 'income_stmt' in data and 'balance_sheet' in data:
                            # Récupérer le bénéfice net
                            net_income = data['income_stmt'].loc['Net Income'].iloc[0]
                            
                            # Récupérer les actifs totaux
                            total_assets = data['balance_sheet'].loc['Total Assets'].iloc[0]
                            
                            if total_assets != 0:
                                roa = net_income / total_assets
                                roa_data[ticker] = roa
                            else:
                                logger.warning(f"Actifs totaux nuls pour {ticker}")
                                roa_data[ticker] = np.nan
                        else:
                            logger.warning(f"Données insuffisantes pour calculer le ROA pour {ticker}")
                            roa_data[ticker] = np.nan
                else:
                    logger.warning(f"Informations fondamentales manquantes pour {ticker}")
                    roa_data[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du ROA pour {ticker}: {e}")
                roa_data[ticker] = np.nan
                
        return pd.DataFrame([roa_data])
        
    @staticmethod
    def calculate_profit_margin(fundamentals):
        """
        Calcule la marge bénéficiaire pour chaque ticker.
        
        Args:
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les marges bénéficiaires
        """
        margin_data = {}
        
        for ticker, data in fundamentals.items():
            try:
                if 'info' in data:
                    # Récupérer la marge directement si disponible
                    profit_margin = data['info'].get('profitMargins', None)
                    
                    if profit_margin is not None:
                        margin_data[ticker] = profit_margin
                    else:
                        # Calculer la marge à partir des états financiers si non disponible directement
                        if 'income_stmt' in data:
                            # Récupérer le bénéfice net
                            net_income = data['income_stmt'].loc['Net Income'].iloc[0]
                            
                            # Récupérer les revenus
                            revenue = data['income_stmt'].loc['Total Revenue'].iloc[0]
                            
                            if revenue != 0:
                                margin = net_income / revenue
                                margin_data[ticker] = margin
                            else:
                                logger.warning(f"Revenus nuls pour {ticker}")
                                margin_data[ticker] = np.nan
                        else:
                            logger.warning(f"Données insuffisantes pour calculer la marge pour {ticker}")
                            margin_data[ticker] = np.nan
                else:
                    logger.warning(f"Informations fondamentales manquantes pour {ticker}")
                    margin_data[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul de la marge pour {ticker}: {e}")
                margin_data[ticker] = np.nan
                
        return pd.DataFrame([margin_data])
        
    @staticmethod
    def calculate_debt_to_equity(fundamentals):
        """
        Calcule le ratio dette/capitaux propres pour chaque ticker.
        
        Args:
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les ratios dette/capitaux propres
        """
        debt_to_equity_data = {}
        
        for ticker, data in fundamentals.items():
            try:
                if 'info' in data:
                    # Récupérer le ratio directement si disponible
                    debt_to_equity = data['info'].get('debtToEquity', None)
                    
                    if debt_to_equity is not None:
                        debt_to_equity_data[ticker] = debt_to_equity
                    else:
                        # Calculer le ratio à partir des états financiers si non disponible directement
                        if 'balance_sheet' in data:
                            # Récupérer la dette totale (court terme + long terme)
                            if 'Total Debt' in data['balance_sheet'].index:
                                total_debt = data['balance_sheet'].loc['Total Debt'].iloc[0]
                            else:
                                short_term_debt = data['balance_sheet'].loc.get('Short Long Term Debt', 0).iloc[0]
                                long_term_debt = data['balance_sheet'].loc.get('Long Term Debt', 0).iloc[0]
                                total_debt = short_term_debt + long_term_debt
                            
                            # Récupérer les capitaux propres
                            equity = data['balance_sheet'].loc['Total Stockholder Equity'].iloc[0]
                            
                            if equity != 0:
                                debt_to_equity = total_debt / equity
                                debt_to_equity_data[ticker] = debt_to_equity
                            else:
                                logger.warning(f"Capitaux propres nuls pour {ticker}")
                                debt_to_equity_data[ticker] = np.nan
                        else:
                            logger.warning(f"Données insuffisantes pour calculer le ratio dette/capitaux propres pour {ticker}")
                            debt_to_equity_data[ticker] = np.nan
                else:
                    logger.warning(f"Informations fondamentales manquantes pour {ticker}")
                    debt_to_equity_data[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du ratio dette/capitaux propres pour {ticker}: {e}")
                debt_to_equity_data[ticker] = np.nan
                
        return pd.DataFrame([debt_to_equity_data])
        
    @staticmethod
    def calculate_interest_coverage(fundamentals):
        """
        Calcule le ratio de couverture des intérêts pour chaque ticker.
        
        Args:
            fundamentals (dict): Dictionnaire contenant les données fondamentales par ticker
            
        Returns:
            pandas.DataFrame: DataFrame contenant les ratios de couverture des intérêts
        """
        interest_coverage_data = {}
        
        for ticker, data in fundamentals.items():
            try:
                if 'income_stmt' in data:
                    # Récupérer l'EBIT (bénéfice avant intérêts et impôts)
                    if 'EBIT' in data['income_stmt'].index:
                        ebit = data['income_stmt'].loc['EBIT'].iloc[0]
                    else:
                        # Calculer l'EBIT si non disponible directement
                        operating_income = data['income_stmt'].loc.get('Operating Income', None).iloc[0]
                        ebit = operating_income
                    
                    # Récupérer les frais d'intérêts
                    if 'Interest Expense' in data['income_stmt'].index:
                        interest_expense = abs(data['income_stmt'].loc['Interest Expense'].iloc[0])
                    else:
                        logger.warning(f"Frais d'intérêts non disponibles pour {ticker}")
                        interest_coverage_data[ticker] = np.nan
                        continue
                    
                    if interest_expense != 0:
                        interest_coverage = ebit / interest_expense
                        interest_coverage_data[ticker] = interest_coverage
                    else:
                        logger.info(f"Frais d'intérêts nuls pour {ticker}, ratio infini")
                        interest_coverage_data[ticker] = np.inf
                else:
                    logger.warning(f"Données de compte de résultat manquantes pour {ticker}")
                    interest_coverage_data[ticker] = np.nan
                    
            except Exception as e:
                logger.error(f"Erreur lors du calcul du ratio de couverture des intérêts pour {ticker}: {e}")
                interest_coverage_data[ticker] = np.nan
                
        return pd.DataFrame([interest_coverage_data])
        
    @staticmethod
    def combine_quality_factors(roe, roa, profit_margin, debt_to_equity, interest_coverage=None, weights=None):
        """
        Combine les différents facteurs de qualité en un score composite.
        
        Args:
            roe (pandas.DataFrame): Return on Equity
            roa (pandas.DataFrame): Return on Assets
            profit_margin (pandas.DataFrame): Marge bénéficiaire
            debt_to_equity (pandas.DataFrame): Ratio dette/capitaux propres
            interest_coverage (pandas.DataFrame, optional): Ratio de couverture des intérêts
            weights (dict, optional): Poids à appliquer à chaque facteur
            
        Returns:
            pandas.DataFrame: Score de qualité composite
        """
        # Poids par défaut
        if weights is None:
            weights = {
                'roe': 0.25, 
                'roa': 0.25, 
                'profit_margin': 0.2, 
                'debt_to_equity': 0.2, 
                'interest_coverage': 0.1
            }
            
        # Normaliser les facteurs
        factors = {}
        
        # Pour ROE, ROA, profit_margin et interest_coverage, plus grand est meilleur
        # Pour debt_to_equity, plus petit est meilleur
        for name, data, reverse in [
            ('roe', roe, False),
            ('roa', roa, False),
            ('profit_margin', profit_margin, False),
            ('debt_to_equity', debt_to_equity, True),
            ('interest_coverage', interest_coverage, False)
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
                
                # Inverser si nécessaire (pour debt_to_equity, plus petit est meilleur)
                if reverse:
                    normalized = -normalized
                
                factors[name] = normalized
        
        # Combiner avec les poids
        composite_score = pd.DataFrame(0, index=roe.index, columns=roe.columns)
        
        for name, factor in factors.items():
            if name in weights and weights[name] > 0:
                composite_score += factor * weights[name]
                
        return composite_score
