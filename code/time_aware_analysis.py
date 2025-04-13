import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib plots
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Starting time-aware analysis for JPX Tokyo Stock Exchange prediction...")

# Define paths to data files
TRAIN_PATH = 'train_files/'
stock_prices_path = os.path.join(TRAIN_PATH, 'stock_prices.csv')
trades_path = os.path.join(TRAIN_PATH, 'trades.csv')
financials_path = os.path.join(TRAIN_PATH, 'financials.csv')
stock_list_path = 'stock_list.csv'

# Load stock list data (smaller file)
print("Loading stock list data...")
stock_list = pd.read_csv(stock_list_path)
print(f"Stock list shape: {stock_list.shape}")

# Function for time-aware sampling
def load_time_aware_sample(filepath, date_column='Date', sample_size=500000, chunks=10):
    """
    Load a sample of data with even time distribution
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    date_column : str
        Name of the date column
    sample_size : int
        Total number of rows to sample
    chunks : int
        Number of time periods to divide the data into
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with samples evenly distributed across time
    """
    print(f"Performing time-aware sampling from {filepath}...")
    
    # First pass: determine date range by reading just the date column
    dates_only = pd.read_csv(filepath, usecols=[date_column])
    dates_only[date_column] = pd.to_datetime(dates_only[date_column])
    
    min_date = dates_only[date_column].min()
    max_date = dates_only[date_column].max()
    date_range = max_date - min_date
    chunk_size = date_range / chunks
    
    print(f"Data spans from {min_date.date()} to {max_date.date()} ({date_range.days} days)")
    
    # Calculate rows per chunk
    rows_per_chunk = sample_size // chunks
    
    # Second pass: sample from each time chunk
    sampled_data = []
    
    for i in range(chunks):
        chunk_start = min_date + i * chunk_size
        chunk_end = min_date + (i + 1) * chunk_size
        
        if i == chunks - 1:  # Make sure we include the max date in the last chunk
            chunk_end = max_date + timedelta(days=1)
            
        print(f"Sampling from period {i+1}/{chunks}: {chunk_start.date()} to {chunk_end.date()}")
        
        # Read only rows within this date range
        chunk_data = pd.read_csv(filepath)
        chunk_data[date_column] = pd.to_datetime(chunk_data[date_column])
        chunk_data = chunk_data[(chunk_data[date_column] >= chunk_start) & 
                               (chunk_data[date_column] < chunk_end)]
        
        # Random sample from this chunk
        if len(chunk_data) > rows_per_chunk:
            chunk_sample = chunk_data.sample(rows_per_chunk, random_state=42+i)
        else:
            chunk_sample = chunk_data  # If chunk has fewer rows, take all of them
            
        sampled_data.append(chunk_sample)
    
    # Combine all chunks
    combined_sample = pd.concat(sampled_data, ignore_index=True)
    print(f"Time-aware sampling complete. Sample shape: {combined_sample.shape}")
    
    return combined_sample

# Load a time-aware sample of stock prices
stock_prices_sample = load_time_aware_sample(stock_prices_path, 'Date', sample_size=500000, chunks=10)
print(f"Stock prices time-aware sample shape: {stock_prices_sample.shape}")

# Basic statistics and distributions
print("\nStock prices data info:")
print(stock_prices_sample.info())

# Verify time distribution
plt.figure(figsize=(14, 6))
stock_prices_sample['Date'] = pd.to_datetime(stock_prices_sample['Date'])
date_counts = stock_prices_sample['Date'].dt.to_period('M').value_counts().sort_index()
date_counts.index = date_counts.index.astype(str)
plt.bar(date_counts.index, date_counts.values)
plt.title('Distribution of Sampled Data Across Time', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Records', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('time_distribution_of_samples.png')
plt.close()

# Visualization 1: Rolling Correlation Analysis Between Stocks
print("\nAnalyzing rolling correlations between major stocks...")

# Rather than using value_counts, let's select stocks with the most complete data
stock_coverage = stock_prices_sample.groupby('SecuritiesCode').size().sort_values(ascending=False)
major_stocks = stock_coverage.head(10).index.tolist()
print(f"Selected stocks with most complete data: {major_stocks}")

# Create a time series of daily returns
stock_prices_sample = stock_prices_sample.sort_values(['SecuritiesCode', 'Date'])
stock_prices_sample['PrevClose'] = stock_prices_sample.groupby('SecuritiesCode')['Close'].shift(1)
stock_prices_sample['DailyReturn'] = (stock_prices_sample['Close'] - stock_prices_sample['PrevClose']) / stock_prices_sample['PrevClose']

# Create pivot table
pivot_returns = stock_prices_sample[stock_prices_sample['SecuritiesCode'].isin(major_stocks)].pivot_table(
    index='Date', 
    columns='SecuritiesCode', 
    values='DailyReturn'
)

# Check data coverage
data_counts = pivot_returns.count()
print(f"Number of data points for each stock: {data_counts.to_dict()}")

# Fill missing values for better correlation calculation (forward fill then backward fill)
pivot_returns = pivot_returns.ffill().bfill()

# Calculate rolling correlations for pairs of stocks
fig, ax = plt.subplots(figsize=(14, 10))
colors = ['b', 'g', 'r', 'c', 'm']

# Pick a reference stock (one with good coverage)
reference_stock = major_stocks[0]
window_size = 30  # Reduced from 60 to 30 days for better coverage

# Define a list of stocks to compare with
comparison_stocks = major_stocks[1:6]

# Create a rolling window correlation function with min_periods
for i, stock in enumerate(comparison_stocks):
    if stock == reference_stock:
        continue
    
    # Use a minimum number of periods (half the window) to allow for some missing data
    rolling_corr = pivot_returns[reference_stock].rolling(
        window=window_size, 
        min_periods=max(5, window_size//3)
    ).corr(pivot_returns[stock])
    
    ax.plot(rolling_corr.index, rolling_corr.values, 
            label=f'Correlation: Stock {reference_stock} vs {stock}', 
            color=colors[i % len(colors)])

ax.set_title(f'{window_size}-Day Rolling Correlation with Stock {reference_stock}', fontsize=16)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Correlation Coefficient', fontsize=14)
ax.legend(loc='best')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_ylim(-0.6, 0.6)  # Set reasonable y-axis limits for correlations
plt.tight_layout()
plt.savefig('rolling_correlations.png')
plt.close()

# Visualization 2: Seasonal Patterns in Stock Returns
print("\nAnalyzing seasonal patterns in stock returns...")

# Extract month and day of week
stock_prices_sample['Month'] = stock_prices_sample['Date'].dt.month
stock_prices_sample['DayOfWeek'] = stock_prices_sample['Date'].dt.dayofweek
stock_prices_sample['Year'] = stock_prices_sample['Date'].dt.year

# Monthly pattern of returns
monthly_returns = stock_prices_sample.groupby('Month')['Target'].mean() * 100  # Convert to percentage

plt.figure(figsize=(12, 6))
monthly_returns.plot(kind='bar', color='skyblue')
plt.title('Average Stock Returns by Month (Seasonal Pattern)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Return (%)', fontsize=14)
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('monthly_return_pattern.png')
plt.close()

# Day of week pattern
daily_returns = stock_prices_sample.groupby('DayOfWeek')['Target'].mean() * 100  # Convert to percentage

plt.figure(figsize=(10, 6))
daily_returns.plot(kind='bar', color='lightgreen')
plt.title('Average Stock Returns by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Average Return (%)', fontsize=14)
plt.xticks(range(5), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('day_of_week_return_pattern.png')
plt.close()

# Visualization 3: Year-to-Year Evolution of Returns
print("\nAnalyzing year-to-year evolution of stock returns...")

# Calculate yearly returns by sector
# Merge with sector data
stock_with_sector = pd.merge(
    stock_prices_sample,
    stock_list[['SecuritiesCode', '33SectorName']],
    on='SecuritiesCode',
    how='left'
)

# Filter to major sectors only (top 10 by count)
major_sectors = stock_with_sector['33SectorName'].value_counts().head(10).index.tolist()
stock_with_sector = stock_with_sector[stock_with_sector['33SectorName'].isin(major_sectors)]

# Calculate yearly returns by sector
yearly_sector_returns = stock_with_sector.groupby(['Year', '33SectorName'])['Target'].mean() * 100
yearly_sector_returns = yearly_sector_returns.reset_index()

# Pivot for plotting
yearly_sector_pivot = yearly_sector_returns.pivot(index='Year', columns='33SectorName', values='Target')

plt.figure(figsize=(16, 10))
yearly_sector_pivot.plot(marker='o', linewidth=2)
plt.title('Year-to-Year Evolution of Sector Returns', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Return (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title='Sector', loc='center left', bbox_to_anchor=(1, 0.5))
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('yearly_sector_returns.png')
plt.close()

# Visualization 4: Return Volatility Relationship Over Time
print("\nAnalyzing return-volatility relationship over time...")

# Calculate monthly returns and volatility
stock_prices_sample['YearMonth'] = stock_prices_sample['Date'].dt.to_period('M')
monthly_metrics = stock_prices_sample.groupby(['YearMonth', 'SecuritiesCode']).agg({
    'DailyReturn': ['mean', 'std']
}).reset_index()

monthly_metrics.columns = ['YearMonth', 'SecuritiesCode', 'MeanReturn', 'ReturnVolatility']
monthly_metrics = monthly_metrics.dropna()

# Calculate average monthly relationship
monthly_corr = monthly_metrics.groupby('YearMonth').apply(
    lambda x: x['MeanReturn'].corr(x['ReturnVolatility'])
).reset_index()
monthly_corr.columns = ['YearMonth', 'ReturnVolatilityCorr']
monthly_corr['YearMonth'] = monthly_corr['YearMonth'].astype(str)

plt.figure(figsize=(14, 6))
plt.bar(monthly_corr['YearMonth'], monthly_corr['ReturnVolatilityCorr'], color='purple', alpha=0.7)
plt.title('Monthly Correlation Between Return and Volatility', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xticks(rotation=90)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('return_volatility_correlation.png')
plt.close()

# Visualization 5: Market Events and Anomaly Detection
print("\nDetecting market anomalies and unusual events...")

# Calculate market-wide metrics
daily_market = stock_prices_sample.groupby('Date').agg({
    'Target': 'mean',
    'DailyReturn': ['mean', 'std', 'count']
}).reset_index()

daily_market.columns = ['Date', 'TargetReturn', 'MeanReturn', 'ReturnStd', 'StockCount']
daily_market = daily_market.sort_values('Date')

# Calculate z-scores for returns
daily_market['ReturnZScore'] = (daily_market['TargetReturn'] - daily_market['TargetReturn'].mean()) / daily_market['TargetReturn'].std()

# Identify anomalies (days with absolute z-score > 2)
anomalies = daily_market[abs(daily_market['ReturnZScore']) > 2].copy()
anomalies['Color'] = anomalies['ReturnZScore'].apply(lambda x: 'red' if x < 0 else 'green')

plt.figure(figsize=(16, 8))
plt.plot(daily_market['Date'], daily_market['TargetReturn'], color='blue', alpha=0.7)
plt.scatter(anomalies['Date'], anomalies['TargetReturn'], 
           color=anomalies['Color'], s=50, zorder=5)

plt.title('Market Returns with Anomaly Detection', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Market Average Return', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('market_anomalies.png')
plt.close()

# Print anomaly dates
print("\nDetected market anomaly dates:")
for _, row in anomalies.iterrows():
    direction = "negative" if row['ReturnZScore'] < 0 else "positive"
    print(f"{row['Date'].date()}: {direction} anomaly, z-score: {row['ReturnZScore']:.2f}, return: {row['TargetReturn']*100:.2f}%")

# Visualization 6: Temporal Analysis of Trading Volume vs Price Movement
print("\nAnalyzing relationship between trading volume and price movement over time...")

# Load trading data
trades_data = pd.read_csv(trades_path)
trades_data['StartDate'] = pd.to_datetime(trades_data['StartDate'])
trades_data['EndDate'] = pd.to_datetime(trades_data['EndDate'])

# Create a time series of weekly trading volume and returns
weekly_volume = trades_data.groupby('EndDate')['TotalTotal'].sum().reset_index()
weekly_volume.columns = ['Date', 'Volume']

# Calculate weekly market returns
weekly_returns = stock_prices_sample.groupby('Date')['Target'].mean().reset_index()
weekly_returns.columns = ['Date', 'Return']

# Merge volume and returns data
volume_returns = pd.merge_asof(
    weekly_volume.sort_values('Date'), 
    weekly_returns.sort_values('Date'),
    on='Date',
    direction='nearest'
)

# Calculate rolling correlation
volume_returns['RollingCorr'] = (
    volume_returns['Volume'].rolling(window=12).corr(volume_returns['Return'])
)

# Plot the evolving relationship
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Volume and returns on first axis
ax1.plot(volume_returns['Date'], volume_returns['Return'], color='blue', label='Return')
ax1.set_ylabel('Return', color='blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')

ax1b = ax1.twinx()
ax1b.plot(volume_returns['Date'], volume_returns['Volume'] / 1e9, color='red', label='Volume (Billions)')
ax1b.set_ylabel('Volume (Billions)', color='red', fontsize=14)
ax1b.tick_params(axis='y', labelcolor='red')

# Add legend for first plot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Rolling correlation on second axis
ax2.plot(volume_returns['Date'], volume_returns['RollingCorr'], color='purple')
ax2.set_ylabel('Rolling Correlation', fontsize=14)
ax2.set_xlabel('Date', fontsize=14)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.suptitle('Relationship Between Trading Volume and Market Returns Over Time', fontsize=16)
plt.tight_layout()
plt.savefig('volume_return_relationship.png')
plt.close()

# Summary of findings
print("\nTime-Aware Analysis Summary:")
print("1. We've ensured proper time coverage in our data sampling")
print("2. Analyzed rolling correlations between major stocks over time")
print("3. Identified seasonal patterns in stock returns (monthly and day-of-week effects)")
print("4. Tracked year-to-year evolution of sector returns")
print("5. Examined the changing relationship between return and volatility")
print("6. Detected market anomalies and unusual events")
print("7. Analyzed the temporal relationship between trading volume and price movements")

print("\nKey findings:")
print("- Stock correlations vary over time, with periods of both high and low correlation")
print("- Returns show distinct seasonal patterns across months and days of the week")
print("- Different sectors show varying performance trends over the years")
print("- The relationship between return and volatility changes over time")
print("- Market anomalies can be detected and may correspond to significant events")
print("- The correlation between trading volume and returns evolves over time")

print("\nNext steps:")
print("1. Incorporate time-based features in prediction models")
print("2. Develop trading strategies that account for seasonal patterns")
print("3. Consider regime-switching models to adapt to changing market conditions")
print("4. Track and respond to anomalies in real-time during model deployment")
print("5. Create sector-specific models that account for their unique temporal characteristics")

print("\nTime-aware analysis completed. Visualization files have been saved to the current directory.") 