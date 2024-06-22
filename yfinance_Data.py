import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data(ticker):
    # Fetching data
    data = yf.download(ticker, period='max', interval='1d')
    
    # Calculate basic features
    data['OrderFlow'] = (data['Close'] - data['Open']) / data['Open'] * 100  # Example calculation
    data['BidAskSpread'] = data['High'] - data['Low']  # Simplified bid-ask spread
    
    # Calculate moving averages
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_30'] = data['Close'].rolling(window=30).mean()
    
    # Calculate exponential moving averages
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
    
    # Calculate MACD
    data['MACD_line'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_signal'] = data['MACD_line'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    data['RSI'] = compute_rsi(data['Close'], 14)  # RSI over 14 periods
    
    # Calculate Volatility
    data['Volatility'] = data['Close'].rolling(window=14).std()  # Rolling volatility over 14 periods

    # Calculate 10-day moving average of Volatility
    data['MA_10_volatility'] = data['Volatility'].rolling(window=10).mean()
    
    # Calculate EMA10 of the MA10 of Volatility
    data['EMA_10_MA_10_volatility'] = data['MA_10_volatility'].ewm(span=10, adjust=False).mean()

    # Set 'StockPrice' as the closing price
    data['StockPrice'] = data['Close']  # Target variable

    # Drop rows with NaN values that may have been created by rolling/ewm functions
    data = data.dropna()

    # Calculate average volatility and total volume
    average_volatility = data['Volatility'].mean()
    total_volume = data['Volume'].sum()

    return data, total_volume, average_volatility

def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def save_to_csv(data, filename):
    data.to_csv(filename)
    print(f"Data saved to {filename}")

def main():
    tickers = [
        'CL=F',  # ,'Enter more', 'Tickers'
    ]


    summary = {}  # Dictionary to store volumes and volatilities

    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        try:
            data, total_volume, average_volatility = fetch_stock_data(ticker)
            filename = f'{ticker}_data.csv'
            save_to_csv(data, filename)
            summary[ticker] = (total_volume, average_volatility)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # Print the volume and volatility summary for each ticker
    for ticker, (volume, volatility) in summary.items():
        print(f"Total volume for {ticker}: {volume}")
        print(f"Average volatility for {ticker}: {volatility}")

if __name__ == '__main__':
    main()



