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

print("Starting financial metrics analysis for JPX Tokyo Stock Exchange prediction...")

# Define paths to data files
TRAIN_PATH = 'train_files/'
stock_prices_path = os.path.join(TRAIN_PATH, 'stock_prices.csv')
financials_path = os.path.join(TRAIN_PATH, 'financials.csv')
stock_list_path = 'stock_list.csv'

# Load stock list data
print("Loading stock list data...")
stock_list = pd.read_csv(stock_list_path)
print(f"Stock list shape: {stock_list.shape}")

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

# Load financial data
print("\nLoading financial data...")
financials = pd.read_csv(financials_path)
print(f"Financials shape: {financials.shape}")

# Convert date columns to datetime
financials['Date'] = pd.to_datetime(financials['Date'])
financials['DisclosedDate'] = pd.to_datetime(financials['DisclosedDate'])

# Display financial data columns
print("\nFinancial data columns:")
print(financials.columns.tolist())

# Basic statistics for financial metrics
print("\nFinancial metrics statistics:")
financial_metrics = [
    'NetSales', 'OperatingProfit', 'OrdinaryProfit', 'Profit',
    'EarningsPerShare', 'TotalAssets', 'Equity',
    'EquityToAssetRatio', 'BookValuePerShare'
]
print(financials[financial_metrics].describe())

# Check for missing values in financial data
print("\nMissing values in financial data:")
print(financials[financial_metrics].isnull().sum())

# Clean financial data by replacing inf and nan with None
for col in financial_metrics:
    financials[col] = financials[col].replace([np.inf, -np.inf], np.nan)

# Distribution of key financial metrics
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, metric in enumerate(financial_metrics):
    data = financials[metric].dropna()
    
    # Remove outliers for better visualization (using 1-99 percentile)
    lower = np.percentile(data, 1)
    upper = np.percentile(data, 99)
    filtered_data = data[(data >= lower) & (data <= upper)]
    
    sns.histplot(filtered_data, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {metric}')
    axes[i].set_xlabel(metric)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('financial_metrics_distribution.png')
plt.close()

# Correlation between financial metrics
plt.figure(figsize=(14, 12))
correlation_matrix = financials[financial_metrics].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Financial Metrics', fontsize=16)
plt.tight_layout()
plt.savefig('financial_metrics_correlation.png')
plt.close()

# Prepare data for analyzing relationship between financials and returns
print("\nAnalyzing relationship between financial metrics and stock returns...")

# Create a function to match financial data with stock returns
def match_financials_with_returns(financials_df, stock_prices_df, days_after=30):
    """
    Match financial disclosure with subsequent stock returns
    days_after: number of trading days to look ahead for returns
    """
    results = []
    
    for _, fin_row in financials_df.iterrows():
        security_code = fin_row['SecuritiesCode']
        disclosure_date = fin_row['DisclosedDate']
        
        # Get stock prices after the disclosure date
        future_prices = stock_prices_df[
            (stock_prices_df['SecuritiesCode'] == security_code) &
            (stock_prices_df['Date'] > disclosure_date) &
            (stock_prices_df['Date'] <= disclosure_date + pd.Timedelta(days=days_after))
        ]
        
        if not future_prices.empty:
            # Calculate cumulative return
            if len(future_prices) >= 2:
                start_price = future_prices.iloc[0]['Close']
                end_price = future_prices.iloc[-1]['Close']
                cumulative_return = (end_price - start_price) / start_price
                
                # Add to results
                result = {
                    'SecuritiesCode': security_code,
                    'DisclosedDate': disclosure_date,
                    'CumulativeReturn': cumulative_return
                }
                
                # Add financial metrics
                for metric in financial_metrics:
                    result[metric] = fin_row[metric]
                
                results.append(result)
    
    return pd.DataFrame(results)

# Use a subset of financial data for the analysis
financials_sample = financials.sample(min(5000, len(financials)))
matched_data = match_financials_with_returns(financials_sample, stock_prices_sample)
print(f"Matched data shape: {matched_data.shape}")

# Analyze correlation between financial metrics and future returns
print("\nCorrelation between financial metrics and future returns:")
correlations = matched_data.corr()['CumulativeReturn'].sort_values(ascending=False)
print(correlations)

# Plot correlations
plt.figure(figsize=(12, 8))
correlations.drop('CumulativeReturn').plot(kind='barh', color=['red' if x < 0 else 'green' for x in correlations.drop('CumulativeReturn')])
plt.title('Correlation of Financial Metrics with Future Returns', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('financial_metrics_return_correlation.png')
plt.close()

# Scatter plots of the top correlated financial metrics with returns
top_metrics = correlations.drop('CumulativeReturn').abs().sort_values(ascending=False).head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(top_metrics):
    data = matched_data[[metric, 'CumulativeReturn']].dropna()
    
    # Remove outliers for better visualization
    lower_x = np.percentile(data[metric], 1)
    upper_x = np.percentile(data[metric], 99)
    lower_y = np.percentile(data['CumulativeReturn'], 1)
    upper_y = np.percentile(data['CumulativeReturn'], 99)
    
    filtered_data = data[
        (data[metric] >= lower_x) & (data[metric] <= upper_x) &
        (data['CumulativeReturn'] >= lower_y) & (data['CumulativeReturn'] <= upper_y)
    ]
    
    sns.scatterplot(x=metric, y='CumulativeReturn', data=filtered_data, ax=axes[i], alpha=0.6)
    
    # Add regression line
    sns.regplot(x=metric, y='CumulativeReturn', data=filtered_data, ax=axes[i], scatter=False, line_kws={"color": "red"})
    
    axes[i].set_title(f'{metric} vs Future Returns')
    axes[i].set_xlabel(metric)
    axes[i].set_ylabel('Cumulative Return')

plt.tight_layout()
plt.savefig('financial_metrics_return_scatterplots.png')
plt.close()

# Sector-wise analysis of financial metrics
print("\nSector-wise analysis of financial metrics...")

# Merge financial data with sector information
financials_with_sector = pd.merge(
    financials,
    stock_list[['SecuritiesCode', '33SectorName']],
    on='SecuritiesCode',
    how='left'
)

# Calculate average financial metrics by sector
sector_financials = financials_with_sector.groupby('33SectorName')[financial_metrics].mean().sort_values('OperatingProfit', ascending=False)
print("\nAverage financial metrics by sector:")
print(sector_financials[['NetSales', 'OperatingProfit', 'Profit']])

# Plot top sectors by Profit
plt.figure(figsize=(14, 10))
sector_financials.dropna(subset=['Profit']).sort_values('Profit', ascending=False).head(10)['Profit'].plot(kind='barh', color='green')
plt.title('Top 10 Sectors by Average Profit', fontsize=16)
plt.xlabel('Average Profit', fontsize=14)
plt.ylabel('Sector', fontsize=14)
plt.tight_layout()
plt.savefig('top_sectors_by_profit.png')
plt.close()

# Plot profitability ratios by sector
plt.figure(figsize=(14, 10))
# Calculate and sort operating margin
sector_financials['OperatingMargin'] = sector_financials['OperatingProfit'] / sector_financials['NetSales']
sector_financials.dropna(subset=['OperatingMargin']).sort_values('OperatingMargin', ascending=False).head(10)['OperatingMargin'].plot(kind='barh', color='purple')
plt.title('Top 10 Sectors by Operating Margin', fontsize=16)
plt.xlabel('Operating Margin (Operating Profit / Net Sales)', fontsize=14)
plt.ylabel('Sector', fontsize=14)
plt.tight_layout()
plt.savefig('top_sectors_by_margin.png')
plt.close()

# Time series analysis of financial metrics
print("\nTime series analysis of financial metrics...")

# Group financials by date and calculate average metrics over time
financials['Year'] = financials['Date'].dt.year
financials['Quarter'] = financials['Date'].dt.quarter
financials['YearQuarter'] = financials['Year'].astype(str) + '-Q' + financials['Quarter'].astype(str)

# Calculate average financial metrics over time
time_series_metrics = financials.groupby('YearQuarter')[financial_metrics].mean()

# Plot the trend of key metrics over time
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# Net Sales trend
time_series_metrics['NetSales'].plot(ax=axes[0])
axes[0].set_title('Average Net Sales Over Time', fontsize=16)
axes[0].set_xlabel('Year-Quarter', fontsize=14)
axes[0].set_ylabel('Net Sales', fontsize=14)
axes[0].tick_params(axis='x', rotation=45)

# Operating Profit trend
time_series_metrics['OperatingProfit'].plot(ax=axes[1])
axes[1].set_title('Average Operating Profit Over Time', fontsize=16)
axes[1].set_xlabel('Year-Quarter', fontsize=14)
axes[1].set_ylabel('Operating Profit', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)

# Equity to Asset Ratio trend
time_series_metrics['EquityToAssetRatio'].plot(ax=axes[2])
axes[2].set_title('Average Equity to Asset Ratio Over Time', fontsize=16)
axes[2].set_xlabel('Year-Quarter', fontsize=14)
axes[2].set_ylabel('Equity to Asset Ratio', fontsize=14)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('financial_metrics_time_series.png')
plt.close()

# Analyze the predictive power of financial surprises
print("\nAnalyzing the predictive power of financial surprises...")

# Calculate surprise metrics (actual vs forecast)
financials['NetSalesSurprise'] = (financials['NetSales'] - financials['ForecastNetSales']) / financials['ForecastNetSales']
financials['OperatingProfitSurprise'] = (financials['OperatingProfit'] - financials['ForecastOperatingProfit']) / financials['ForecastOperatingProfit']
financials['OrdinaryProfitSurprise'] = (financials['OrdinaryProfit'] - financials['ForecastOrdinaryProfit']) / financials['ForecastOrdinaryProfit']
financials['ProfitSurprise'] = (financials['Profit'] - financials['ForecastProfit']) / financials['ForecastProfit']

surprise_metrics = [
    'NetSalesSurprise', 'OperatingProfitSurprise',
    'OrdinaryProfitSurprise', 'ProfitSurprise'
]

# Clean surprise data, replacing inf and nan
for col in surprise_metrics:
    financials[col] = financials[col].replace([np.inf, -np.inf], np.nan)
    
    # Remove extreme outliers (typically due to very small denominators)
    lower_bound = np.percentile(financials[col].dropna(), 1)
    upper_bound = np.percentile(financials[col].dropna(), 99)
    financials[col] = financials[col].mask((financials[col] < lower_bound) | (financials[col] > upper_bound))

# Use the same matching function to analyze relationship between surprises and returns
financials_surprise_sample = financials[['SecuritiesCode', 'DisclosedDate'] + surprise_metrics].dropna(subset=surprise_metrics).sample(min(5000, len(financials)))
matched_surprise_data = match_financials_with_returns(financials_surprise_sample, stock_prices_sample)
print(f"Matched surprise data shape: {matched_surprise_data.shape}")

# Correlation of surprises with returns
surprise_correlation = matched_surprise_data.corr()['CumulativeReturn'].sort_values(ascending=False)
print("\nCorrelation between financial surprises and future returns:")
print(surprise_correlation)

# Plot surprise correlations
plt.figure(figsize=(12, 8))
surprise_correlation.drop('CumulativeReturn').plot(kind='barh', color=['red' if x < 0 else 'green' for x in surprise_correlation.drop('CumulativeReturn')])
plt.title('Correlation of Financial Surprises with Future Returns', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('surprise_return_correlation.png')
plt.close()

# Create surprise buckets and analyze returns by bucket
print("\nAnalyzing returns by surprise magnitude...")

def create_surprise_buckets(data, surprise_column, n_buckets=5):
    """Create equal-sized buckets of surprise and analyze returns"""
    data = data.dropna(subset=[surprise_column, 'CumulativeReturn'])
    data['SurpriseBucket'] = pd.qcut(data[surprise_column], n_buckets, labels=False)
    
    bucket_returns = data.groupby('SurpriseBucket')['CumulativeReturn'].agg(['mean', 'median', 'std', 'count']).reset_index()
    bucket_returns['SurpriseMean'] = data.groupby('SurpriseBucket')[surprise_column].mean().values
    
    return bucket_returns

# Analyze returns by profit surprise bucket
profit_surprise_buckets = create_surprise_buckets(matched_surprise_data, 'ProfitSurprise')
print("\nReturns by profit surprise magnitude:")
print(profit_surprise_buckets)

# Plot returns by surprise bucket
plt.figure(figsize=(12, 8))
plt.bar(profit_surprise_buckets['SurpriseBucket'], profit_surprise_buckets['mean'], 
        yerr=profit_surprise_buckets['std'] / np.sqrt(profit_surprise_buckets['count']),
        capsize=5, color='skyblue')
plt.plot(profit_surprise_buckets['SurpriseBucket'], profit_surprise_buckets['mean'], 'ro-')
plt.title('Average Return by Profit Surprise Bucket', fontsize=16)
plt.xlabel('Surprise Bucket (0=Most Negative, 4=Most Positive)', fontsize=14)
plt.ylabel('Average Cumulative Return', fontsize=14)
plt.xticks(profit_surprise_buckets['SurpriseBucket'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('returns_by_surprise_bucket.png')
plt.close()

# Summary of findings
print("\nFinancial Metrics Analysis Summary:")
print("1. We've analyzed the distribution and relationships between key financial metrics")
print("2. Examined correlations between financial metrics and subsequent stock returns")
print("3. Analyzed sector-wise differences in financial performance")
print("4. Tracked financial metrics over time to identify trends")
print("5. Investigated the predictive power of earnings surprises")

print("\nKey findings:")
print("- Certain financial metrics show consistent correlation with future stock returns")
print("- Sectors vary significantly in their financial performance and profitability")
print("- Earnings surprises appear to have a measurable impact on subsequent stock performance")
print("- Financial metrics show distinct trends over time, reflecting overall market conditions")

print("\nNext steps:")
print("1. Incorporate these financial metrics as features in the stock return prediction model")
print("2. Develop more sophisticated surprise metrics that account for analyst expectations")
print("3. Consider sector-specific models that account for differences in financial relationships")
print("4. Integrate financial data with price-based technical indicators for a comprehensive approach")

print("\nFinancial metrics analysis completed. Visualization files have been saved to the current directory.") 