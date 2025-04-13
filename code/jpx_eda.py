import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib plots
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Define paths to data files
TRAIN_PATH = 'train_files/'
stock_prices_path = os.path.join(TRAIN_PATH, 'stock_prices.csv')
secondary_stock_prices_path = os.path.join(TRAIN_PATH, 'secondary_stock_prices.csv')
trades_path = os.path.join(TRAIN_PATH, 'trades.csv')
options_path = os.path.join(TRAIN_PATH, 'options.csv')
financials_path = os.path.join(TRAIN_PATH, 'financials.csv')
stock_list_path = 'stock_list.csv'

print("Starting exploratory data analysis for JPX Tokyo Stock Exchange prediction...")

# Load stock list data (smaller file)
print("Loading stock list data...")
stock_list = pd.read_csv(stock_list_path)
print(f"Stock list shape: {stock_list.shape}")

# Display unique sectors and market segments
sector_counts = stock_list[stock_list['33SectorCode'].notna()]['33SectorName'].value_counts()
print("\nDistribution of stocks by sector:")
print(sector_counts)

market_segment_counts = stock_list[stock_list['NewMarketSegment'].notna()]['NewMarketSegment'].value_counts()
print("\nDistribution of stocks by market segment:")
print(market_segment_counts)

# Plot sector distribution
plt.figure(figsize=(14, 10))
sector_counts.plot(kind='barh', color='skyblue')
plt.title('Number of Stocks by Sector', fontsize=16)
plt.xlabel('Number of Stocks', fontsize=14)
plt.ylabel('Sector', fontsize=14)
plt.tight_layout()
plt.savefig('sector_distribution.png')
plt.close()

# Function to load and sample stock price data (since it's large)
def load_stock_prices_sample(filepath, nrows=None, random_state=42):
    """Load a sample of stock prices to work with manageable data size"""
    if nrows:
        # Load just the first nrows for a quick look
        return pd.read_csv(filepath, nrows=nrows)
    else:
        # For full analysis, read the entire file
        return pd.read_csv(filepath)

# Load a sample of the stock prices data
print("\nLoading a sample of stock prices data...")
stock_prices_sample = load_stock_prices_sample(stock_prices_path, nrows=500000)
print(f"Stock prices sample shape: {stock_prices_sample.shape}")

# Basic statistics and distributions
print("\nStock prices data info:")
stock_prices_sample.info()

print("\nStock prices data statistics:")
print(stock_prices_sample.describe())

# Convert Date column to datetime
stock_prices_sample['Date'] = pd.to_datetime(stock_prices_sample['Date'])

# Check for missing values
print("\nMissing values in stock prices data:")
print(stock_prices_sample.isnull().sum())

# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(stock_prices_sample['Target'], kde=True, bins=100)
plt.title('Distribution of Target Variable (Returns)', fontsize=16)
plt.xlabel('Return', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xlim(-0.1, 0.1)  # Focus on the main distribution range
plt.savefig('target_distribution.png')
plt.close()

# Time series analysis for a few selected stocks
print("\nPerforming time series analysis on selected stocks...")

# Select a few stocks to analyze
sample_securities = stock_prices_sample['SecuritiesCode'].value_counts().head(5).index.tolist()

# Create time series plot
plt.figure(figsize=(14, 8))
for security in sample_securities:
    security_data = stock_prices_sample[stock_prices_sample['SecuritiesCode'] == security].sort_values('Date')
    plt.plot(security_data['Date'], security_data['Close'], label=f'Security {security}')

plt.title('Stock Price Movement for Selected Securities', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Closing Price', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('stock_price_movement.png')
plt.close()

# Analyze volume trends
plt.figure(figsize=(14, 8))
for security in sample_securities:
    security_data = stock_prices_sample[stock_prices_sample['SecuritiesCode'] == security].sort_values('Date')
    plt.plot(security_data['Date'], security_data['Volume'], label=f'Security {security}')

plt.title('Trading Volume for Selected Securities', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Volume', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('trading_volume.png')
plt.close()

# Volatility analysis
print("\nCalculating daily returns and volatility...")
# Group by security code and date, then calculate daily returns
stock_prices_sample = stock_prices_sample.sort_values(['SecuritiesCode', 'Date'])
stock_prices_sample['PrevClose'] = stock_prices_sample.groupby('SecuritiesCode')['Close'].shift(1)
stock_prices_sample['DailyReturn'] = (stock_prices_sample['Close'] - stock_prices_sample['PrevClose']) / stock_prices_sample['PrevClose']

# Calculate volatility (standard deviation of returns over a window)
# Here we use a 20-day rolling window for each security
stock_prices_sample['Volatility_20d'] = stock_prices_sample.groupby('SecuritiesCode')['DailyReturn'].transform(
    lambda x: x.rolling(window=20, min_periods=5).std()
)

# Plot volatility for selected securities
plt.figure(figsize=(14, 8))
for security in sample_securities:
    security_data = stock_prices_sample[stock_prices_sample['SecuritiesCode'] == security].sort_values('Date')
    plt.plot(security_data['Date'], security_data['Volatility_20d'], label=f'Security {security}')

plt.title('20-Day Volatility for Selected Securities', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Volatility (Std Dev of Returns)', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('volatility.png')
plt.close()

# Correlation between stocks
print("\nAnalyzing correlations between stocks...")

# Create a pivot table of daily returns
pivot_returns = stock_prices_sample.pivot_table(
    index='Date', 
    columns='SecuritiesCode', 
    values='DailyReturn'
)

# Calculate correlation matrix on a subset of stocks
correlation_matrix = pivot_returns.iloc[:, :20].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Stock Returns', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Load and analyze trades data
print("\nLoading trading data...")
trades_data = pd.read_csv(trades_path)
trades_data['StartDate'] = pd.to_datetime(trades_data['StartDate'])
trades_data['EndDate'] = pd.to_datetime(trades_data['EndDate'])
print(f"Trades data shape: {trades_data.shape}")

# Aggregate trading behavior by investor type
investor_columns = [col for col in trades_data.columns if 'Balance' in col]
investor_balance = trades_data[investor_columns].sum().sort_values()

plt.figure(figsize=(12, 8))
investor_balance.plot(kind='barh', color=['red' if x < 0 else 'green' for x in investor_balance])
plt.title('Net Trading Position by Investor Type', fontsize=16)
plt.xlabel('Net Position (Buy-Sell)', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('investor_trading_behavior.png')
plt.close()

# Analyze trading volume trends
plt.figure(figsize=(14, 8))
trades_data.groupby('EndDate')['TotalTotal'].sum().plot()
plt.title('Weekly Trading Volume Trend', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Trading Volume', fontsize=14)
plt.tight_layout()
plt.savefig('trading_volume_trend.png')
plt.close()

# Sector performance analysis
print("\nAnalyzing sector performance...")

# Merge stock price data with stock list to get sector information
stock_with_sector = pd.merge(
    stock_prices_sample,
    stock_list[['SecuritiesCode', '33SectorName', '17SectorName']],
    on='SecuritiesCode',
    how='left'
)

# Calculate average returns by sector
sector_returns = stock_with_sector.groupby('33SectorName')['Target'].mean().sort_values()

plt.figure(figsize=(14, 10))
sector_returns.plot(kind='barh', color=['red' if x < 0 else 'green' for x in sector_returns])
plt.title('Average Stock Returns by Sector', fontsize=16)
plt.xlabel('Average Return', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('sector_returns.png')
plt.close()

# Feature importance for predicting returns
print("\nAnalyzing feature importance for predicting returns...")

# Create some basic features from the stock price data
stock_prices_sample['PriceRange'] = stock_prices_sample['High'] - stock_prices_sample['Low']
stock_prices_sample['PriceRangeRatio'] = stock_prices_sample['PriceRange'] / stock_prices_sample['Open']
stock_prices_sample['ClosingGap'] = (stock_prices_sample['Close'] - stock_prices_sample['Open']) / stock_prices_sample['Open']

# Calculate rolling means and ratios
stock_prices_sample['MA5'] = stock_prices_sample.groupby('SecuritiesCode')['Close'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
stock_prices_sample['MA20'] = stock_prices_sample.groupby('SecuritiesCode')['Close'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)
stock_prices_sample['CloseToMA5Ratio'] = stock_prices_sample['Close'] / stock_prices_sample['MA5']
stock_prices_sample['CloseToMA20Ratio'] = stock_prices_sample['Close'] / stock_prices_sample['MA20']

# Calculate volume features
stock_prices_sample['VolumeMA5'] = stock_prices_sample.groupby('SecuritiesCode')['Volume'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
stock_prices_sample['VolumeRatio'] = stock_prices_sample['Volume'] / stock_prices_sample['VolumeMA5']

# Correlation with target
feature_columns = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'PrevClose', 'DailyReturn', 'Volatility_20d',
    'PriceRange', 'PriceRangeRatio', 'ClosingGap',
    'MA5', 'MA20', 'CloseToMA5Ratio', 'CloseToMA20Ratio',
    'VolumeMA5', 'VolumeRatio'
]

correlations = stock_prices_sample[feature_columns + ['Target']].corr()['Target'].sort_values(ascending=False)

plt.figure(figsize=(12, 10))
correlations.drop('Target').plot(kind='barh', color=['red' if x < 0 else 'blue' for x in correlations.drop('Target')])
plt.title('Feature Correlation with Target Return', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Analysis of market trends over time
print("\nAnalyzing market trends over time...")

# Calculate market-wide average returns by date
market_returns = stock_prices_sample.groupby('Date')['Target'].mean()

plt.figure(figsize=(14, 8))
market_returns.plot()
plt.title('Market Average Returns Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Average Return', fontsize=14)
plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('market_returns.png')
plt.close()

# Market-wide volatility over time
market_volatility = stock_prices_sample.groupby('Date')['Volatility_20d'].mean()

plt.figure(figsize=(14, 8))
market_volatility.plot()
plt.title('Market-Wide Volatility Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Average 20-Day Volatility', fontsize=14)
plt.tight_layout()
plt.savefig('market_volatility.png')
plt.close()

# Summary and recommendations
print("\nEDA Summary:")
print("1. We've analyzed the distribution of stocks across sectors and market segments")
print("2. Examined stock price movements, trading volumes, and volatility for selected securities")
print("3. Analyzed correlations between stocks and identified trends")
print("4. Investigated trading patterns by different investor types")
print("5. Compared sector performance in terms of average returns")
print("6. Identified features with the strongest correlation to future returns")
print("7. Analyzed market-wide trends in returns and volatility")

print("\nNext steps for modeling:")
print("1. Engineer additional features based on technical indicators")
print("2. Incorporate fundamental data from the financials.csv file")
print("3. Consider market sentiment from options data")
print("4. Develop time-series forecasting models for predicting future returns")
print("5. Evaluate models using appropriate metrics for financial time series")

print("\nEDA completed. Visualization files have been saved to the current directory.") 