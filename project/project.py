import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TO-DO: 
# değerler doğru mu fonksiyonlar istenildiği gibi çalışıyor mu, kontrol edilcek. 
# gördüğüm kadarıyla pnl ve calculate_annualized_return düzgün çalışmıyor. 
# optional: Transaction Costs
# optional: Additional Cointegration Tests (Johansen Test):
# good to have: Explore multi-asset trading (statistical arbitrage across multiple cointegrated assets):

# Function to get data
def get_data(pair, start='2014-01-01', end='2024-01-01'):
    data = yf.download(pair, start=start, end=end, progress=False)['Close']
    data = data.fillna(method='ffill')  # Forward fill missing data
    return data.astype(float)

# Function to test cointegration
# cointegration test using the Engle-Granger method
def test_cointegration(series1, series2):
    score, p_value, _ = coint(series1, series2)
    return p_value

# Augmented Dickey-Fuller test
def adf_test(series):
    series = series.dropna()
    if len(series) == 0:
        return 1.0  # Return high p-value if series is empty
    series = series.values.flatten()  # Ensure it's a 1D array
    if np.all(series == series[0]):
        return 1.0  # Return high p-value if series is constant
    result = adfuller(series)
    return result[1]  # p-value

# Function to calculate spread
def calculate_spread(asset1, asset2):

    aligned_data = pd.concat([asset1, asset2], axis=1, join='inner')

    asset1_aligned = aligned_data.iloc[:, 0]
    asset2_aligned = aligned_data.iloc[:, 1]
    
    # Run linear regression to find hedge ratio
    model = sm.OLS(asset1_aligned, sm.add_constant(asset2_aligned)).fit()
    hedge_ratio = model.params[1]

    # Calculate spread
    spread = asset1_aligned - hedge_ratio * asset2_aligned
    return spread


# Function to calculate Z-score
def calculate_zscore(spread):
    return (spread - spread.mean()) / spread.std()

# Function to plot each pair in its own plot
def plot_pair(asset1, asset2):
    
    plt.figure(figsize=(12, 8))
    plt.plot(asset1.index, asset1, label=f'{pair[0]}')
    plt.plot(asset2.index, asset2, label=f'{pair[1]}')
    plt.title(f'Closing Prices: {pair[0]} vs {pair[1]}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Function to implement pair trading strategy
def pair_trading_strategy(asset1, asset2):
    spread = calculate_spread(asset1, asset2)
    z_score = calculate_zscore(spread)
    
    plt.figure(figsize=(12,6))
    plt.plot(z_score.index, z_score, label='Z-Score of Spread')
    plt.axhline(2, color='r', linestyle='--', label='Upper Threshold')
    plt.axhline(-2, color='g', linestyle='--', label='Lower Threshold')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.title("Pair Trading Strategy - Z Score")
    plt.show()
    
    return z_score.dropna().astype(float)

# Function to backtest trading strategy
def backtest(z_scores, entry_threshold=1.5, exit_threshold=0.5): # change 1, -1 to more frequent trades
    positions = []
    for z in z_scores.dropna():
        if z > entry_threshold:
            positions.append(-1)  # Short spread
        elif z < -entry_threshold:
            positions.append(1)  # Long spread
        elif abs(z) < exit_threshold:
            positions.append(0)  # Close position
        else:
            positions.append(positions[-1] if positions else 0)  # Hold
    return positions

# Function to calculate profit and loss (PnL)
def calculate_pnl(positions, asset1, asset2):
    # Align the data based on their indices (dates)
    aligned_data = pd.concat([asset1, asset2], axis=1, join='inner')
    asset1_aligned = aligned_data.iloc[:, 0]
    asset2_aligned = aligned_data.iloc[:, 1]
    
    pnl = []
    for i in range(1, len(positions)):
        if positions[i-1] == 1:  # Long position
            pnl.append(asset1_aligned[i] - asset2_aligned[i])
        elif positions[i-1] == -1:  # Short position
            pnl.append(asset2_aligned[i] - asset1_aligned[i])
        else:
            pnl.append(0)  # No position
    return np.array(pnl)

def calculate_annualized_return(pnl, period_in_years):
    if len(pnl) == 0 or np.sum(pnl) == 0:
        return np.nan  # Return NaN if there are no trades or no PnL

    cumulative_return = np.sum(pnl) / len(pnl)  # Calculate average PnL per trade
    annualized_return = (1 + cumulative_return) ** period_in_years - 1  # Compounding over the time period
    return annualized_return

# Function to evaluate performance metrics
def evaluate_performance(pnl, risk_free_rate=0.01):
    # Net profit
    total_pnl = np.sum(pnl)
    print(f"Total PnL: {total_pnl}")

    # Sharpe ratio
    excess_returns = pnl - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized Sharpe ratio
    print(f"Sharpe Ratio: {sharpe_ratio}")

    # Maximum drawdown
    cumulative_pnl = np.cumsum(pnl)
    drawdowns = cumulative_pnl - np.maximum.accumulate(cumulative_pnl)
    max_drawdown = np.min(drawdowns)
    print(f"Maximum Drawdown: {max_drawdown}")

    # Win rate
    win_rate = len([x for x in pnl if x > 0]) / len(pnl) * 100
    print(f"Win Rate: {win_rate}%")

    # Annualized return
    years = len(pnl) / 252  # Assuming 252 trading days in a year
    annualized_return = calculate_annualized_return(pnl, years)
    print(f"Annualized Return: {annualized_return}")

def plot_cumulative_pnl(pnl):
    cumulative_pnl = np.cumsum(pnl)
    plt.figure(figsize=(12,6))
    plt.plot(cumulative_pnl, label='Cumulative PnL')
    plt.title('Cumulative PnL Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.show()


# List of stock pairs
stock_pairs = [('AAPL', 'MSFT'), ('AU', 'AG'), ('NVDA', 'AMD'), ('JPM', 'GS'), ('KO', 'PEP'), ('SPY', 'QQQ')]

# List of currency pairs
currency_pairs = [('AUDUSD=X', 'CADUSD=X')]

multi_assets = [
    ('WMT', 'TGT'), 
    ('WMT', 'COST'), 
    ('TGT', 'COST'), 

    ('MSFT', 'AAPL'), 
    ('MSFT', 'GOOGL'), 
    ('AAPL', 'GOOGL'), 

    ('AUDUSD=X', 'USDCAD=X'), 
    ('AUDUSD=X', 'NZDUSD=X'), 
    ('USDCAD=X', 'NZDUSD=X'), 

    ('EURUSD=X', 'GBPUSD=X'), 
    ('EURUSD=X', 'USDJPY=X'), 
    ('GBPUSD=X', 'USDJPY=X')
]


# Combine the stock and currency pairs
pairs = stock_pairs + currency_pairs

# Create an empty dictionary to store the data
pair_data = {}

# Download the data for each pair
for pair in pairs:
    asset1 = get_data(pair[0])
    asset2 = get_data(pair[1])
    """print(asset1.describe())
    print(asset2.describe())

    print(len(asset1))
    print(len(asset2))"""
    pair_data[pair] = (asset1, asset2)

    plot_pair(asset1, asset2)
    
    # Test for cointegration
    p_value = test_cointegration(asset1, asset2)
    print(f"Cointegration p-value for {pair[0]} and {pair[1]}: {p_value:.4f}")
    
    # Compute and test spread
    spread = calculate_spread(asset1, asset2)
    print(f"spread for {pair[0]} and {pair[1]}: {spread}")
    adf_p_value = adf_test(spread)
    print(f"ADF Test p-value for {pair[0]} and {pair[1]}: {adf_p_value:.4f}")
    
    # Run pair trading strategy
    z_scores = pair_trading_strategy(asset1, asset2)
    print(f"z_scores for {pair[0]} and {pair[1]}: {z_scores}")
    positions = backtest(z_scores)
    print(f"Backtesting complete for {pair[0]} and {pair[1]}. Sample positions: {positions[:10]}")
    
    positions = backtest(z_scores, 1, -1)
    print(f"Backtesting complete for {pair[0]} and {pair[1]}. Sample positions: {positions[:10]}")

    # Calculate PnL and evaluate performance
    pnl = calculate_pnl(positions, asset1, asset2)
    evaluate_performance(pnl)

    plot_cumulative_pnl(pnl)

    print("\n")
