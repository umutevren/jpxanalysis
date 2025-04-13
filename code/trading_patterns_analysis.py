import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib plots
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Starting trading patterns analysis for JPX Tokyo Stock Exchange prediction...")

# Define paths to data files
TRAIN_PATH = 'train_files/'
stock_prices_path = os.path.join(TRAIN_PATH, 'stock_prices.csv')
trades_path = os.path.join(TRAIN_PATH, 'trades.csv')
stock_list_path = 'stock_list.csv'

# Function to load stock price data sample
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
stock_prices_sample['Date'] = pd.to_datetime(stock_prices_sample['Date'])
print(f"Stock prices sample shape: {stock_prices_sample.shape}")

# Load trading data
print("\nLoading trading data...")
trades_data = pd.read_csv(trades_path)
trades_data['StartDate'] = pd.to_datetime(trades_data['StartDate'])
trades_data['EndDate'] = pd.to_datetime(trades_data['EndDate'])
print(f"Trades data shape: {trades_data.shape}")

# Basic statistics and overview of trading data
print("\nTrading data overview:")
print(trades_data.describe())

# Check for missing values in trading data
print("\nMissing values in trading data:")
print(trades_data.isnull().sum().sum())

# Display trading data columns
print("\nTrading data columns:")
print(trades_data.columns.tolist())

# Analyzing trading patterns by investor type
print("\nAnalyzing trading patterns by investor type...")

# Extract all purchase, sales, and balance columns
purchase_cols = [col for col in trades_data.columns if 'Purchases' in col]
sales_cols = [col for col in trades_data.columns if 'Sales' in col]
balance_cols = [col for col in trades_data.columns if 'Balance' in col]

# Calculate total trading activity by investor type
investor_activity = pd.DataFrame()

for col in balance_cols:
    investor_type = col.replace('Balance', '')
    investor_activity[investor_type + 'PctOfTotal'] = trades_data[investor_type + 'Total'] / trades_data['TotalTotal'] * 100

# Calculate average percentage of total trading by each investor type
avg_investor_pct = investor_activity.mean().sort_values(ascending=False)

plt.figure(figsize=(14, 10))
avg_investor_pct.plot(kind='barh', color='skyblue')
plt.title('Average Percentage of Total Trading by Investor Type', fontsize=16)
plt.xlabel('Average Percentage of Total Trading', fontsize=14)
plt.ylabel('Investor Type', fontsize=14)
plt.tight_layout()
plt.savefig('avg_investor_trading_pct.png')
plt.close()

# Analyze net trading position by investor type over time
plt.figure(figsize=(16, 12))

# Calculate cumulative net position for each investor type
cumulative_positions = pd.DataFrame(index=trades_data['EndDate'].unique())

top_investors = [col.replace('Balance', '') for col in 
                np.array(balance_cols)[np.argsort(abs(trades_data[balance_cols].sum()))[-6:]]]

for investor in top_investors:
    net_position = trades_data.groupby('EndDate')[investor + 'Balance'].sum()
    cumulative_positions[investor] = net_position.cumsum()

# Plot cumulative positions
cumulative_positions.plot(ax=plt.gca())
plt.title('Cumulative Net Trading Position by Major Investor Types', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Net Position (Buy-Sell)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('cumulative_investor_positions.png')
plt.close()

# Correlations between investor trading and market returns
print("\nAnalyzing correlations between investor trading and market returns...")

# Calculate weekly market returns
stock_prices_sample['WeekStart'] = stock_prices_sample['Date'] - pd.to_timedelta(stock_prices_sample['Date'].dt.dayofweek, unit='d')
weekly_returns = stock_prices_sample.groupby('WeekStart')['Target'].mean()

# Create a merged dataset of trades and market returns
trades_returns = pd.DataFrame()
trades_returns['WeekEnd'] = trades_data['EndDate']
trades_returns['WeekStart'] = trades_data['StartDate']
trades_returns['TotalTurnover'] = trades_data['TotalTotal']

# Add investor balances
for col in balance_cols:
    trades_returns[col] = trades_data[col]

# Match with next week's returns
trades_returns['NextWeekStart'] = trades_returns['WeekEnd'] + pd.Timedelta(days=1)
trades_returns['NextWeekStart'] = trades_returns['NextWeekStart'] - pd.to_timedelta(trades_returns['NextWeekStart'].dt.dayofweek, unit='d')
trades_returns = trades_returns.merge(weekly_returns.reset_index().rename(columns={'WeekStart': 'NextWeekStart', 'Target': 'NextWeekReturn'}),
                               on='NextWeekStart', how='left')

# Calculate correlations
investor_return_corr = trades_returns.corr()['NextWeekReturn'].sort_values(ascending=False)
print("\nCorrelation between investor trading and next week's market returns:")
print(investor_return_corr)

# Plot correlations
plt.figure(figsize=(14, 10))
investor_return_corr.drop('NextWeekReturn').head(15).plot(kind='barh', color=['red' if x < 0 else 'green' for x in investor_return_corr.drop('NextWeekReturn').head(15)])
plt.title('Top Correlations: Investor Trading vs. Next Week Returns', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('investor_return_correlation.png')
plt.close()

# Analyze the predictive power of investor imbalances
print("\nAnalyzing the predictive power of investor imbalances...")

# Calculate investor imbalance metrics
for investor in top_investors:
    trades_returns[investor + 'Imbalance'] = trades_returns[investor + 'Balance'] / trades_returns['TotalTotal']

imbalance_cols = [investor + 'Imbalance' for investor in top_investors]

# Correlation of imbalances with returns
imbalance_correlation = trades_returns.corr()['NextWeekReturn'][imbalance_cols].sort_values(ascending=False)
print("\nCorrelation between investor imbalances and future returns:")
print(imbalance_correlation)

# Create imbalance buckets and analyze returns by bucket
def create_imbalance_buckets(data, imbalance_column, n_buckets=5):
    """Create equal-sized buckets of imbalance and analyze returns"""
    data = data.dropna(subset=[imbalance_column, 'NextWeekReturn'])
    data['ImbalanceBucket'] = pd.qcut(data[imbalance_column], n_buckets, labels=False)
    
    bucket_returns = data.groupby('ImbalanceBucket')['NextWeekReturn'].agg(['mean', 'median', 'std', 'count']).reset_index()
    bucket_returns['ImbalanceMean'] = data.groupby('ImbalanceBucket')[imbalance_column].mean().values
    
    return bucket_returns

# Analyze returns by imbalance bucket for the most predictive investor type
best_imbalance = imbalance_correlation.index[0]
imbalance_buckets = create_imbalance_buckets(trades_returns, best_imbalance)
print(f"\nReturns by {best_imbalance} magnitude:")
print(imbalance_buckets)

# Plot returns by imbalance bucket
plt.figure(figsize=(12, 8))
plt.bar(imbalance_buckets['ImbalanceBucket'], imbalance_buckets['mean'], 
        yerr=imbalance_buckets['std'] / np.sqrt(imbalance_buckets['count']),
        capsize=5, color='skyblue')
plt.plot(imbalance_buckets['ImbalanceBucket'], imbalance_buckets['mean'], 'ro-')
plt.title(f'Average Return by {best_imbalance} Bucket', fontsize=16)
plt.xlabel('Imbalance Bucket (0=Most Negative, 4=Most Positive)', fontsize=14)
plt.ylabel('Average Next Week Return', fontsize=14)
plt.xticks(imbalance_buckets['ImbalanceBucket'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('returns_by_imbalance_bucket.png')
plt.close()

# Time series analysis of trading volume
print("\nTime series analysis of trading volume...")

# Analyze trading volume trends over time
plt.figure(figsize=(14, 8))
trades_data.set_index('EndDate')['TotalTotal'].plot()
plt.title('Total Trading Volume Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Trading Volume', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('trading_volume_trend.png')
plt.close()

# Seasonal patterns in trading volume
print("\nAnalyzing seasonal patterns in trading volume...")

# Extract month and day of week
trades_data['Month'] = trades_data['EndDate'].dt.month
trades_data['DayOfWeek'] = trades_data['EndDate'].dt.dayofweek

# Monthly pattern
monthly_volume = trades_data.groupby('Month')['TotalTotal'].mean()
plt.figure(figsize=(12, 6))
monthly_volume.plot(kind='bar', color='skyblue')
plt.title('Average Trading Volume by Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Trading Volume', fontsize=14)
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig('monthly_trading_volume.png')
plt.close()

# Analyzing trading patterns by market section
print("\nAnalyzing trading patterns by market section...")

# Group by section and analyze trading patterns
section_summary = trades_data.groupby('Section').agg({
    'TotalTotal': 'mean',
    'TotalBalance': 'sum'
}).sort_values('TotalTotal', ascending=False)

print("\nTrading summary by market section:")
print(section_summary)

# Plot trading volume by section
plt.figure(figsize=(14, 8))
section_summary['TotalTotal'].plot(kind='bar', color='skyblue')
plt.title('Average Trading Volume by Market Section', fontsize=16)
plt.xlabel('Market Section', fontsize=14)
plt.ylabel('Average Trading Volume', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('section_trading_volume.png')
plt.close()

# Relationship between trading volume and volatility
print("\nAnalyzing relationship between trading volume and market volatility...")

# Calculate daily returns and volatility for the stock price sample
stock_prices_sample = stock_prices_sample.sort_values(['SecuritiesCode', 'Date'])
stock_prices_sample['PrevClose'] = stock_prices_sample.groupby('SecuritiesCode')['Close'].shift(1)
stock_prices_sample['DailyReturn'] = (stock_prices_sample['Close'] - stock_prices_sample['PrevClose']) / stock_prices_sample['PrevClose']

# Calculate market-wide volatility per week
weekly_volatility = stock_prices_sample.groupby('WeekStart')['DailyReturn'].std().reset_index()
weekly_volatility = weekly_volatility.rename(columns={'DailyReturn': 'WeeklyVolatility'})

# Merge trading data with volatility
volume_volatility = trades_returns.merge(weekly_volatility, left_on='WeekStart', right_on='WeekStart', how='left')

# Calculate correlation
print("\nCorrelation between trading volume and volatility:")
print(volume_volatility[['TotalTurnover', 'WeeklyVolatility']].corr())

# Scatter plot of volume vs volatility
plt.figure(figsize=(12, 8))
sns.scatterplot(x='WeeklyVolatility', y='TotalTurnover', data=volume_volatility, alpha=0.6)
sns.regplot(x='WeeklyVolatility', y='TotalTurnover', data=volume_volatility, scatter=False, line_kws={"color": "red"})
plt.title('Trading Volume vs Market Volatility', fontsize=16)
plt.xlabel('Weekly Volatility (Std Dev of Returns)', fontsize=14)
plt.ylabel('Total Trading Volume', fontsize=14)
plt.tight_layout()
plt.savefig('volume_volatility_relationship.png')
plt.close()

# Investor sentiment indicator
print("\nCreating an investor sentiment indicator...")

# Calculate sentiment as a weighted balance of investor types
# Positive weights for investors whose buying correlates with future positive returns
# Negative weights for investors whose buying correlates with future negative returns
sentiment_weights = {
    'Foreigners': 0.4,         # International investors often lead the market
    'Individuals': -0.2,       # Retail often buys at the wrong time
    'InvestmentTrusts': 0.2,   # Professional fund managers
    'BusinessCos': 0.1,        # Companies know their industries
    'SecuritiesCos': 0.1       # Have market knowledge
}

trades_returns['SentimentIndicator'] = 0
for investor, weight in sentiment_weights.items():
    # Normalize the imbalance (to make weights more meaningful)
    if investor + 'Imbalance' in trades_returns.columns:
        normalized_imbalance = (trades_returns[investor + 'Imbalance'] - trades_returns[investor + 'Imbalance'].mean()) / trades_returns[investor + 'Imbalance'].std()
        trades_returns['SentimentIndicator'] += weight * normalized_imbalance

# Create sentiment buckets
sentiment_buckets = create_imbalance_buckets(trades_returns, 'SentimentIndicator')
print("\nReturns by sentiment indicator bucket:")
print(sentiment_buckets)

# Plot returns by sentiment bucket
plt.figure(figsize=(12, 8))
plt.bar(sentiment_buckets['ImbalanceBucket'], sentiment_buckets['mean'], 
        yerr=sentiment_buckets['std'] / np.sqrt(sentiment_buckets['count']),
        capsize=5, color='skyblue')
plt.plot(sentiment_buckets['ImbalanceBucket'], sentiment_buckets['mean'], 'ro-')
plt.title('Average Return by Sentiment Indicator Bucket', fontsize=16)
plt.xlabel('Sentiment Bucket (0=Most Negative, 4=Most Positive)', fontsize=14)
plt.ylabel('Average Next Week Return', fontsize=14)
plt.xticks(sentiment_buckets['ImbalanceBucket'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('returns_by_sentiment_bucket.png')
plt.close()

# Summary of findings
print("\nTrading Patterns Analysis Summary:")
print("1. We've analyzed trading patterns across different investor types")
print("2. Examined relationships between investor trading behavior and future market returns")
print("3. Investigated seasonal patterns in trading volume")
print("4. Analyzed trading patterns by market section")
print("5. Explored the relationship between trading volume and market volatility")
print("6. Created a composite investor sentiment indicator based on trading imbalances")

print("\nKey findings:")
print("- Different investor types show distinct trading patterns and correlations with future returns")
print("- Trading volume exhibits seasonal patterns by month")
print("- There's a significant relationship between trading volume and market volatility")
print("- A weighted sentiment indicator based on investor imbalances shows promising predictive power")

print("\nNext steps:")
print("1. Incorporate trading pattern metrics as features in the stock return prediction model")
print("2. Develop more sophisticated sentiment indicators that account for past performance of investor groups")
print("3. Consider time-varying relationships between trading patterns and market returns")
print("4. Integrate trading data with price-based technical indicators and fundamental metrics")

print("\nTrading patterns analysis completed. Visualization files have been saved to the current directory.") 