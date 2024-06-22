###################################################
# Machine Learning Stock Trade
# by alex lassaline
####################################################

# Note: This code uses API's, ensure you Pip the imports in CMD before running

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

graph_option = int(input("G_O: "))

def load_data(filename, directory, skiprows=None):
    print(f"Loading data from {filename}...")
    df = pd.read_csv(os.path.join(directory, filename), header=0, skiprows=skiprows)
    df['Ticker'] = filename.replace('_data.csv', '')
    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('Unknown')
    return df

def initialize_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=50000, random_state=43, min_samples_leaf=9))
    ])
    return pipeline

def iterative_walk_forward_validation_and_trading(df, features, initial_train_size, initial_capital=10000):
    if df.empty or len(df) < initial_train_size:
        print(f"Skipping due to insufficient data. DataFrame is empty or smaller than initial train size: {len(df)}")
        return None

    train_df = df.iloc[:initial_train_size]
    test_df = df.iloc[initial_train_size:]
    if train_df.empty:
        print("No data available for training. Check initial train size settings.")
        return None

    trading_sessions = []
    ml_cash_values = []
    commodity_cash_values = []
    daily_volatilities = []
    initial_commodity_price = test_df['StockPrice'].iloc[0]

    cash = initial_capital
    commodity_cash = initial_capital

    old_price = None
    older_price = 0
    position = None
    total_volatility = 0
    total_volume = 0

    pipeline = initialize_model()
    X_train = train_df[features]
    y_train = train_df['StockPrice']
    pipeline.fit(X_train, y_train)

    for index, row in test_df.iterrows():
        X_test = pd.DataFrame([row[features]], columns=features)
        y_pred = pipeline.predict(X_test)[0]

        if old_price is not None:
            shares_traded = cash // old_price
            if position == 0:
                profit = shares_traded * (row['StockPrice'] - old_price) * 1
                cash += profit
            if position == 1:
                profit = shares_traded * (row['StockPrice'] - old_price) * 2
                cash += profit


        if y_pred> row['StockPrice']:
            position = 1
            old_price = row['StockPrice']
        else:
            position = 0
            old_price = row['StockPrice']



        trading_sessions.append(index)
        ml_cash_values.append(cash)
        current_commodity_value = commodity_cash * (row['StockPrice'] / initial_commodity_price)
        commodity_cash_values.append(current_commodity_value)
        daily_volatilities.append(row['Volatility'])

        total_volatility += row['Volatility']
        total_volume += row['Volume']

    mse = mean_squared_error(test_df['StockPrice'], pipeline.predict(test_df[features]))
    r2 = r2_score(test_df['StockPrice'], pipeline.predict(test_df[features]))

    return {
        'mse': mse,
        'r2': r2,
        'final_cash': cash,
        'commodity_final': current_commodity_value,
        'avg_volatility': total_volatility / len(test_df),
        'avg_volume': total_volume / len(test_df),
        'sessions': trading_sessions,
        'ml_values': ml_cash_values,
        'commodity_values': commodity_cash_values,
        'volatilities': daily_volatilities
    }

def plot_performance_comparison(trading_sessions, ml_cash_values, commodity_cash_values, volatilities, title):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Trade Sessions')
    ax1.set_ylabel('Cash Value', color=color)
    ax1.plot(trading_sessions, ml_cash_values, label='ML Trading Cash Over Time', color=color)
    ax1.plot(trading_sessions, commodity_cash_values, label='Commodity Investment Cash Over Time', color='orange')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Volatility', color=color)  # we already handled the x-label with ax1
    ax2.plot(trading_sessions, volatilities, label='Daily Volatility', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_volatility_vs_cash(trading_sessions, volatility_changes, cash_changes):
    plt.figure(figsize=(12, 6))
    plt.plot(trading_sessions, volatility_changes, label='Change in Volatility (20-trade Rolling Avg)', color='red')
    plt.plot(trading_sessions, cash_changes, label='Change in Cash (20-trade Rolling Avg)', color='green')
    plt.xlabel('Trade Sessions')
    plt.ylabel('% Change')
    plt.title('Change in Volatility and Cash Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

directory = '../Basic Machine Learning'
features = ['MA_10', 'MA_30', 'EMA_10', 'EMA_30', 'MACD_line', 'MACD_signal', 'RSI', 'Volatility', 'MA_10_volatility', 'EMA_10_MA_10_volatility', 'StockPrice']
results_list = []

for filename in os.listdir(directory):
    if filename.endswith('_data.csv'):
        df = load_data(filename, directory)
        initial_train_size = int(len(df) * 0.8)
        result = iterative_walk_forward_validation_and_trading(df, features, initial_train_size)
        if result:
            results_list.append(result)
            print(f"Results for {filename}: MSE = {result['mse']:.3f}, R² = {result['r2']:.3f}, Final Cash = ${result['final_cash']:.2f}, Avg. Volatility = {result['avg_volatility']:.3f}, Avg. Volume = {result['avg_volume']:.0f}")
            if graph_option == 1:
                plot_performance_comparison(result['sessions'], result['ml_values'], result['commodity_values'], result['volatilities'], f"Performance of {filename.replace('_data.csv', '')}")

for result in results_list:
    print(f"MSE: {result['mse']}, R²: {result['r2']}, Final Cash: {result['final_cash']}, Avg. Volatility: {result['avg_volatility']}, Avg. Volume: {result['avg_volume']}")
