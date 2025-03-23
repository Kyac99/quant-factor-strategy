"""
Module pour récupérer les données financières nécessaires à la stratégie.
"""
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Classe pour charger les données financières depuis diverses sources.
    """
    
    def __init__(self, start_date=None, end_date=None, data_dir='data'):
        """
        Initialise le DataLoader.
        
        Args:
            start_date (str): Date de début au format 'YYYY-MM-DD'
            end_date (str): Date de fin au format 'YYYY-MM-DD'
            data_dir (str): Répertoire où stocker les données
        """
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = data_dir
        
        # Créer le répertoire de données s'il n'existe pas
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
    def get_stock_data(self, ticker_list, interval='1d', save=True):
        """
        Récupère les données de cours pour une liste de tickers.
        
        Args:
            ticker_list (list): Liste des symboles boursiers
            interval (str): Intervalle de temps ('1d', '1wk', '1mo', etc.)
            save (bool): Sauvegarder les données dans un fichier CSV
            
        Returns:
            dict: Dictionnaire de DataFrames, une pour chaque ticker
        """
        data = {}
        
        for ticker in ticker_list:
            try:
                logger.info(f"Récupération des données pour {ticker}")
                stock_data = yf.download(
                    ticker, 
                    start=self.start_date, 
                    end=self.end_date,
                    interval=interval,
                    progress=False
                )
                
                # Ne garder que les données OHLCV et Adj Close
                stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                
                if not stock_data.empty:
                    data[ticker] = stock_data
                    
                    # Sauvegarder dans un fichier CSV
                    if save:
                        filename = os.path.join(self.data_dir, f"{ticker}_prices_{interval}.csv")
                        stock_data.to_csv(filename)
                        logger.info(f"Données sauvegardées dans {filename}")
                else:
                    logger.warning(f"Pas de données disponibles pour {ticker}")
                    
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données pour {ticker}: {e}")
                
        return data
        
    def get_fundamentals(self, ticker_list, save=True):
        """
        Récupère les données fondamentales pour une liste de tickers.
        
        Args:
            ticker_list (list): Liste des symboles boursiers
            save (bool): Sauvegarder les données dans un fichier CSV
            
        Returns:
            dict: Dictionnaire de DataFrames, une pour chaque ticker
        """
        fundamentals = {}
        
        for ticker in ticker_list:
            try:
                logger.info(f"Récupération des fondamentaux pour {ticker}")
                ticker_obj = yf.Ticker(ticker)
                
                # Récupérer les informations générales
                info = ticker_obj.info
                
                # Récupérer les états financiers
                balance_sheet = ticker_obj.balance_sheet
                income_stmt = ticker_obj.income_stmt
                cash_flow = ticker_obj.cashflow
                
                # Stocker toutes les données
                fundamentals[ticker] = {
                    'info': info,
                    'balance_sheet': balance_sheet,
                    'income_stmt': income_stmt,
                    'cash_flow': cash_flow
                }
                
                # Sauvegarder dans des fichiers CSV
                if save:
                    # Créer un sous-répertoire pour ce ticker
                    ticker_dir = os.path.join(self.data_dir, ticker)
                    if not os.path.exists(ticker_dir):
                        os.makedirs(ticker_dir)
                    
                    # Sauvegarder les états financiers
                    balance_sheet.to_csv(os.path.join(ticker_dir, 'balance_sheet.csv'))
                    income_stmt.to_csv(os.path.join(ticker_dir, 'income_stmt.csv'))
                    cash_flow.to_csv(os.path.join(ticker_dir, 'cash_flow.csv'))
                    
                    # Sauvegarder les infos générales
                    pd.DataFrame([info]).to_csv(os.path.join(ticker_dir, 'info.csv'))
                    
                    logger.info(f"Fondamentaux sauvegardés dans {ticker_dir}")
                    
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des fondamentaux pour {ticker}: {e}")
                
        return fundamentals
        
    def get_market_index(self, index_ticker='^GSPC', interval='1d', save=True):
        """
        Récupère les données d'un indice de marché (par défaut S&P 500).
        
        Args:
            index_ticker (str): Symbole de l'indice
            interval (str): Intervalle de temps ('1d', '1wk', '1mo', etc.)
            save (bool): Sauvegarder les données dans un fichier CSV
            
        Returns:
            pandas.DataFrame: Données de l'indice
        """
        try:
            logger.info(f"Récupération des données pour l'indice {index_ticker}")
            index_data = yf.download(
                index_ticker,
                start=self.start_date,
                end=self.end_date,
                interval=interval,
                progress=False
            )
            
            # Sauvegarder dans un fichier CSV
            if save and not index_data.empty:
                filename = os.path.join(self.data_dir, f"{index_ticker.replace('^', '')}_index_{interval}.csv")
                index_data.to_csv(filename)
                logger.info(f"Données de l'indice sauvegardées dans {filename}")
                
            return index_data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour l'indice {index_ticker}: {e}")
            return pd.DataFrame()
            
    @staticmethod
    def combine_prices(price_dict, column='Adj Close'):
        """
        Combine les données de prix de plusieurs tickers en un seul DataFrame.
        
        Args:
            price_dict (dict): Dictionnaire de DataFrames de prix
            column (str): Colonne à extraire (par défaut 'Adj Close')
            
        Returns:
            pandas.DataFrame: DataFrame combiné
        """
        combined = pd.DataFrame()
        
        for ticker, data in price_dict.items():
            if column in data.columns:
                combined[ticker] = data[column]
                
        return combined
