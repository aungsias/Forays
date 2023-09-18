import pandas as pd
import requests

class MarketNews:
    """Fetches market news for a given stock ticker and index.

    Attributes:
        url (str): URL endpoint for news API.
        query (dict): Parameters for API request.
        articles (list): List of fetched articles.
        data (DataFrame): Processed articles in DataFrame format.
        _api_key (str): API key for accessing the news API.
    """
    
    def __init__(self, api_key):
        """Initializes MarketNews with an API key."""
        self.url = None
        self.query = None
        self.articles = None
        self.data = None
        self._api_key = api_key
    
    def get_news(self, index, ticker, base_url, start, end, offset=0, limit=10000):
        """Fetches news articles and stores them in the `data` attribute.

        Args:
            index (str): Stock index (e.g., 'NASDAQ').
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            base_url (str): Base URL for the news API.
            start (str): Start date for news articles.
            end (str): End date for news articles.
            offset (int, optional): Offset for pagination. Defaults to 0.
            limit (int, optional): Number of articles to fetch. Defaults to 10000.
        
        Returns:
            MarketNews: Returns self to enable method chaining.
        """
        self.url = self._load_url(index, ticker, base_url)
        self.query = self._build_query(offset, limit, start, end)
        self.articles = self._get_articles(self.url, self.query)
        self.data = self._get_headlines(self.articles, ticker)
        return self

    def _get_headlines(self, articles, ticker):
        """Converts articles into a DataFrame.

        Args:
            articles (list): List of article dictionaries.
            ticker (str): Stock ticker symbol for DataFrame column.

        Returns:
            DataFrame: DataFrame with headlines indexed by publication date.
        """
        return pd.DataFrame.from_dict({
            pd.to_datetime(pd.to_datetime(article["published"]).strftime("%Y-%m-%d %H:%M:%S")): \
                article["headline"] for article in articles
        }, orient='index', columns=[ticker])

    def _get_articles(self, url, params):
        """Fetches articles using the API.

        Args:
            url (str): URL endpoint for news API.
            params (dict): Parameters for API request.

        Returns:
            list: List of fetched articles.
        """
        return requests.get(url, params=params).json()
    
    def _load_url(self, index, ticker, base_url):
        """Constructs the URL for news API.

        Args:
            index (str): Stock index (e.g., 'NASDAQ').
            ticker (str): Stock ticker symbol (e.g., 'AAPL').
            base_url (str): Base URL for the news API.

        Returns:
            str: Constructed URL.
        """
        return f"{base_url}/{index}:{ticker}/news"
    
    def _build_query(self, offset, limit, start, end):
        """Builds query parameters for news API.

        Args:
            offset (int): Offset for pagination.
            limit (int): Number of articles to fetch.
            start (str): Start date for news articles.
            end (str): End date for news articles.

        Returns:
            dict: Query parameters.
        """
        return {
            "offset": str(offset),
            "limit": str(limit),
            "q": "",
            "news_source": "",
            "from": start,
            "to": end,
            "api_token": self._api_key
        }
