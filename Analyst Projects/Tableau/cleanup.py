import yfinance as yf
import pandas as pd
import requests

# Get S&P 500 tickers from Wikipedia
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers).text
sp500 = pd.read_html(html)[0]
sp500 = sp500[['Symbol', 'Security', 'GICS Sector']]
sp500.columns = ['Symbol', 'Name', 'Sector']
sp500.to_csv('constituents.csv', index=False)

# Download price data
tickers = sp500['Symbol'].tolist()

data = yf.download(
    tickers=tickers,
    start='2022-01-01',
    end='2025-01-31',
    interval='1d',
    auto_adjust=True,
    progress=True
)

# Reshape to long format
ohlcv = data[['Open', 'High', 'Low', 'Close', 'Volume']].stack().reset_index()
ohlcv.columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']

# Merge with sector info
final = ohlcv.merge(sp500[['Symbol', 'Sector']], on='Symbol', how='left')
final.dropna(inplace=True)
final.to_csv('sp500_prices_2022_2025.csv', index=False)

print(f"Done. {len(final)} rows exported.")