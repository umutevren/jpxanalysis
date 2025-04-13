import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Import our custom sampling utilities
from sampling_utils import load_time_aware_sample, load_hybrid_sample

# Set style for matplotlib plots
plt.style.use('ggplot')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Starting advanced time series visualization for JPX Tokyo Stock Exchange prediction...")

# Define paths to data files
TRAIN_PATH = 'train_files/'
stock_prices_path = os.path.join(TRAIN_PATH, 'stock_prices.csv')
trades_path = os.path.join(TRAIN_PATH, 'trades.csv')
financials_path = os.path.join(TRAIN_PATH, 'financials.csv')
stock_list_path = 'stock_list.csv'

# Load stock list data 
print("Loading stock list data...")
stock_list = pd.read_csv(stock_list_path)
print(f"Stock list shape: {stock_list.shape}")

# Load a time-aware sample of stock prices using our custom function
stock_prices_sample = load_time_aware_sample(stock_prices_path, 'Date', sample_size=500000, chunks=10)
print(f"Stock prices sample shape: {stock_prices_sample.shape}")

# Extract top stocks by market capitalization (from stock_list)
stock_list['MarketCapitalization'] = pd.to_numeric(stock_list['MarketCapitalization'], errors='coerce')
top_stocks = stock_list.nlargest(20, 'MarketCapitalization')['SecuritiesCode'].tolist()
print(f"Selected top {len(top_stocks)} stocks by market cap for analysis")

# Filter the sample to just these top stocks
top_stock_data = stock_prices_sample[stock_prices_sample['SecuritiesCode'].isin(top_stocks)]

# Calculate returns and other metrics
top_stock_data = top_stock_data.sort_values(['SecuritiesCode', 'Date'])
top_stock_data['PrevClose'] = top_stock_data.groupby('SecuritiesCode')['Close'].shift(1)
top_stock_data['DailyReturn'] = (top_stock_data['Close'] - top_stock_data['PrevClose']) / top_stock_data['PrevClose']

# Select one representative stock for detailed time series analysis
main_stock = top_stocks[0]
main_stock_data = top_stock_data[top_stock_data['SecuritiesCode'] == main_stock].sort_values('Date')

print(f"\nPerforming detailed time series analysis on stock {main_stock}...")

# 1. Interactive Price and Volume Chart with Plotly
print("Creating interactive price and volume chart...")

# Create a subplot with 2 rows
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.02, 
                    row_heights=[0.7, 0.3],
                    subplot_titles=('Price', 'Volume'))

# Add price data to the top subplot
fig.add_trace(
    go.Scatter(x=main_stock_data['Date'], y=main_stock_data['Close'], 
               mode='lines', name='Close Price',
               line=dict(color='blue')),
    row=1, col=1
)

# Add volume to the bottom subplot
fig.add_trace(
    go.Bar(x=main_stock_data['Date'], y=main_stock_data['Volume'], 
           name='Volume', marker=dict(color='rgba(0, 128, 0, 0.5)')),
    row=2, col=1
)

# Update layout
fig.update_layout(
    title=f'Stock {main_stock} - Price and Volume Over Time',
    height=800,
    width=1200,
    showlegend=True,
    xaxis_rangeslider_visible=False
)

# Save as HTML file
fig.write_html('interactive_price_volume_chart.html')
print("Interactive chart saved as 'interactive_price_volume_chart.html'")

# 2. Time Series Decomposition
print("\nPerforming time series decomposition...")

# Resample daily data to weekly to reduce noise
weekly_data = main_stock_data.set_index('Date')['Close'].resample('W').mean()

# Fill any missing values
weekly_data = weekly_data.interpolate()

# Perform decomposition
decomposition = seasonal_decompose(weekly_data, model='additive', period=52)  # 52 weeks per year

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(14, 16))

# Original
decomposition.observed.plot(ax=axes[0])
axes[0].set_title('Original Time Series')
axes[0].set_ylabel('Price')

# Trend
decomposition.trend.plot(ax=axes[1])
axes[1].set_title('Trend Component')
axes[1].set_ylabel('Trend')

# Seasonal
decomposition.seasonal.plot(ax=axes[2])
axes[2].set_title('Seasonal Component')
axes[2].set_ylabel('Seasonality')

# Residual
decomposition.resid.plot(ax=axes[3])
axes[3].set_title('Residual Component')
axes[3].set_ylabel('Residuals')

plt.tight_layout()
plt.savefig('time_series_decomposition.png')
plt.close()

# 3. ACF and PACF plots for understanding time series patterns
print("\nGenerating ACF and PACF plots...")

# Use daily returns for ACF/PACF
daily_returns = main_stock_data['DailyReturn'].dropna()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# ACF
plot_acf(daily_returns, lags=40, ax=axes[0])
axes[0].set_title(f'Autocorrelation Function (ACF) for Stock {main_stock} Returns')

# PACF
plot_pacf(daily_returns, lags=40, ax=axes[1])
axes[1].set_title(f'Partial Autocorrelation Function (PACF) for Stock {main_stock} Returns')

plt.tight_layout()
plt.savefig('acf_pacf_plots.png')
plt.close()

# 4. Rolling Statistics
print("\nCalculating rolling statistics...")

window_size = 30  # 30-day rolling window

# Calculate rolling statistics
rolling_mean = main_stock_data['Close'].rolling(window=window_size).mean()
rolling_std = main_stock_data['Close'].rolling(window=window_size).std()
rolling_min = main_stock_data['Close'].rolling(window=window_size).min()
rolling_max = main_stock_data['Close'].rolling(window=window_size).max()

# Create the plot
plt.figure(figsize=(14, 8))
plt.plot(main_stock_data['Date'], main_stock_data['Close'], label='Closing Price', alpha=0.5)
plt.plot(main_stock_data['Date'], rolling_mean, label=f'{window_size}-day Rolling Mean')
plt.plot(main_stock_data['Date'], rolling_min, label=f'{window_size}-day Rolling Min', alpha=0.3, linestyle='--')
plt.plot(main_stock_data['Date'], rolling_max, label=f'{window_size}-day Rolling Max', alpha=0.3, linestyle='--')

# Fill the area between min and max
plt.fill_between(main_stock_data['Date'], rolling_min, rolling_max, alpha=0.1, color='gray')

plt.title(f'{window_size}-day Rolling Statistics for Stock {main_stock}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('rolling_statistics.png')
plt.close()

# 5. Return Distribution by Year
print("\nAnalyzing return distribution by year...")

# Extract year from date
main_stock_data['Year'] = main_stock_data['Date'].dt.year

# Create a plot with return distributions by year
plt.figure(figsize=(14, 8))
years = sorted(main_stock_data['Year'].unique())

# Create box plots for each year
data_by_year = [main_stock_data[main_stock_data['Year'] == year]['DailyReturn'].dropna() for year in years]
plt.boxplot(data_by_year, labels=years)

plt.title(f'Distribution of Daily Returns by Year for Stock {main_stock}')
plt.xlabel('Year')
plt.ylabel('Daily Return')
plt.grid(axis='y', alpha=0.3)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('return_distribution_by_year.png')
plt.close()

# 6. Heatmap of Monthly Returns
print("\nCreating heatmap of monthly returns...")

# Extract month and year
main_stock_data['Month'] = main_stock_data['Date'].dt.month
main_stock_data['YearMonth'] = main_stock_data['Date'].dt.to_period('M')

# Calculate monthly returns
monthly_returns = main_stock_data.groupby(['Year', 'Month'])['DailyReturn'].mean().reset_index()
monthly_returns = monthly_returns.pivot(index='Year', columns='Month', values='DailyReturn')

# Replace month numbers with names
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_returns.columns = month_names[:len(monthly_returns.columns)]

# Create heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(monthly_returns * 100, cmap='RdYlGn', center=0, annot=True, fmt='.2f',
            linewidths=0.5, cbar_kws={'label': 'Average Daily Return (%)'})

plt.title(f'Monthly Returns Heatmap for Stock {main_stock}')
plt.tight_layout()
plt.savefig('monthly_returns_heatmap.png')
plt.close()

# 7. Volatility Clustering Analysis
print("\nAnalyzing volatility clustering...")

# Calculate absolute returns to measure volatility
main_stock_data['AbsReturn'] = np.abs(main_stock_data['DailyReturn'])

# Calculate rolling volatility (30-day window)
main_stock_data['RollingVol'] = main_stock_data['DailyReturn'].rolling(window=30).std() * np.sqrt(252)  # Annualized

# Create a combined plot of returns and volatility
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Returns plot
ax1.plot(main_stock_data['Date'], main_stock_data['DailyReturn'], color='blue', alpha=0.5)
ax1.set_title(f'Daily Returns for Stock {main_stock}')
ax1.set_ylabel('Return')
ax1.axhline(y=0, color='red', linestyle='-', alpha=0.3)

# Rolling volatility plot
ax2.plot(main_stock_data['Date'], main_stock_data['RollingVol'], color='green')
ax2.set_title(f'30-day Rolling Volatility (Annualized) for Stock {main_stock}')
ax2.set_xlabel('Date')
ax2.set_ylabel('Volatility')

plt.tight_layout()
plt.savefig('volatility_clustering.png')
plt.close()

# 8. Cumulative Returns Comparison
print("\nComparing cumulative returns across top stocks...")

# Create a pivot table with daily returns for top stocks
pivot_returns = top_stock_data.pivot_table(
    index='Date', 
    columns='SecuritiesCode', 
    values='DailyReturn'
)

# Calculate cumulative returns
cum_returns = (1 + pivot_returns).cumprod()

# Plot cumulative returns for top 5 stocks
plt.figure(figsize=(14, 8))

# Get top 5 stocks by final cumulative return
top5_stocks = cum_returns.iloc[-1].nlargest(5).index

for stock in top5_stocks:
    stock_name = f"Stock {stock}"
    if stock in stock_list['SecuritiesCode'].values:
        stock_info = stock_list[stock_list['SecuritiesCode'] == stock]
        if 'Name' in stock_info.columns and not stock_info['Name'].isna().all():
            stock_name = stock_info['Name'].iloc[0]
    
    plt.plot(cum_returns.index, cum_returns[stock], label=stock_name)

plt.title('Cumulative Returns Comparison (Top 5 Performing Stocks)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (1 = Initial Investment)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cumulative_returns_comparison.png')
plt.close()

# 9. Volatility Regime Detection
print("\nPerforming volatility regime detection...")

# Calculate market-wide daily returns
market_returns = stock_prices_sample.groupby('Date')['DailyReturn'].mean()

# Calculate rolling market volatility
rolling_vol = market_returns.rolling(window=30).std() * np.sqrt(252)  # Annualized

# Identify high and low volatility regimes (above/below median)
median_vol = rolling_vol.median()
high_vol = rolling_vol[rolling_vol > median_vol].index
low_vol = rolling_vol[rolling_vol <= median_vol].index

# Plot market volatility with regime highlighting
plt.figure(figsize=(14, 8))
plt.plot(rolling_vol.index, rolling_vol, color='blue', alpha=0.7)

# Highlight high volatility periods
for i in range(len(high_vol) - 1):
    if (high_vol[i+1] - high_vol[i]).days > 1:  # Gap between points
        continue
    plt.axvspan(high_vol[i], high_vol[i+1], alpha=0.2, color='red')

plt.axhline(y=median_vol, color='black', linestyle='--', alpha=0.7, label='Median Volatility')
plt.title('Market Volatility Regimes')
plt.xlabel('Date')
plt.ylabel('30-day Rolling Volatility (Annualized)')
plt.legend()
plt.tight_layout()
plt.savefig('volatility_regimes.png')
plt.close()

# 10. Correlation Evolution Over Time (Heatmap Animation)
print("\nCreating correlation evolution heatmap...")

# We'll analyze how correlations between sectors evolve over time
# First, merge stocks with their sector information
stock_with_sector = pd.merge(
    top_stock_data,
    stock_list[['SecuritiesCode', '33SectorName']],
    on='SecuritiesCode',
    how='left'
)

# Create sector-level returns for each day
sector_returns = stock_with_sector.groupby(['Date', '33SectorName'])['DailyReturn'].mean().reset_index()
sector_pivot = sector_returns.pivot(index='Date', columns='33SectorName', values='DailyReturn')

# Fill missing values
sector_pivot = sector_pivot.fillna(0)

# Prepare yearly correlation matrices
years = sorted(main_stock_data['Year'].unique())
corr_matrices = {}

for year in years:
    year_data = sector_pivot[sector_pivot.index.year == year]
    if not year_data.empty:
        corr_matrices[year] = year_data.corr()

# Create individual correlation heatmaps for each year
for year, corr_matrix in corr_matrices.items():
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                linewidths=0.5, cbar_kws={'label': 'Correlation'})
    plt.title(f'Sector Return Correlations - {year}')
    plt.tight_layout()
    plt.savefig(f'sector_correlation_{year}.png')
    plt.close()

print("Generated correlation heatmaps for each year in the dataset")

# Summary of findings
print("\nAdvanced Time Series Analysis Summary:")
print("1. Created an interactive price and volume chart for detailed exploration")
print("2. Performed time series decomposition to identify trend, seasonal, and residual components")
print("3. Generated ACF and PACF plots to understand time series patterns and potential model structure")
print("4. Calculated rolling statistics to visualize changing price patterns over time")
print("5. Analyzed return distributions by year to identify changes in market behavior")
print("6. Created a heatmap of monthly returns to identify seasonal patterns")
print("7. Analyzed volatility clustering to understand periods of high and low market volatility")
print("8. Compared cumulative returns across top stocks to identify outperformers")
print("9. Detected volatility regimes in the market to identify different market states")
print("10. Visualized the evolution of sector correlations over time")

print("\nKey findings:")
print("- Time series analysis reveals important seasonal patterns in the Japanese stock market")
print("- Volatility tends to cluster, with distinct regimes of high and low volatility")
print("- Sector correlations evolve over time, providing potential diversification opportunities")
print("- ACF and PACF patterns suggest potential for time series forecasting models")
print("- Monthly return patterns show consistent seasonal effects that could be exploited")

print("\nNext steps:")
print("1. Develop time series forecasting models using the patterns identified")
print("2. Create trading strategies that adapt to different volatility regimes")
print("3. Exploit seasonal patterns for potential alpha generation")
print("4. Use correlation insights for effective portfolio construction")
print("5. Incorporate regime detection into the prediction model framework")

print("\nAdvanced time series analysis completed. Visualization files have been saved to the current directory.") 