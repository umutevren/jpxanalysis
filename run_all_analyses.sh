#!/bin/bash

echo "======================================================================"
echo "JPX Tokyo Stock Exchange Prediction - Comprehensive Exploratory Analysis"
echo "======================================================================"

echo ""
echo "This script will run all the exploratory data analyses sequentially."
echo "Please ensure you have all the necessary data files in the correct locations."
echo ""
echo "Starting analyses..."
echo ""

# Install required packages
echo "Installing required packages..."
pip install pandas numpy matplotlib seaborn plotly statsmodels scipy

# Run the time-aware analyses first
echo "1. Running time-aware analysis (time_aware_analysis.py)..."
python time_aware_analysis.py
echo "Time-aware analysis completed."
echo ""

# Run the advanced time series visualizations
echo "2. Running advanced time series visualization (advanced_time_series_viz.py)..."
python advanced_time_series_viz.py
echo "Advanced time series visualization completed."
echo ""

# Run the main EDA script
echo "3. Running main EDA script (jpx_eda.py)..."
python jpx_eda.py
echo "Main EDA completed."
echo ""

# Run the financial metrics analysis
echo "4. Running financial metrics analysis (financial_metrics_analysis.py)..."
python financial_metrics_analysis.py
echo "Financial metrics analysis completed."
echo ""

# Run the trading patterns analysis
echo "5. Running trading patterns analysis (trading_patterns_analysis.py)..."
python trading_patterns_analysis.py
echo "Trading patterns analysis completed."
echo ""

echo "All analyses have been completed successfully."
echo "Visualization files have been saved to the current directory."
echo ""
echo "You can review the following key visualizations for time-aware insights:"
echo "- time_distribution_of_samples.png: Verify even time distribution in samples"
echo "- rolling_correlations.png: See how stock correlations evolve over time"
echo "- monthly_return_pattern.png: Identify seasonal patterns in returns"
echo "- yearly_sector_returns.png: Track sector performance over years"
echo "- market_anomalies.png: Identify unusual market events"
echo "- time_series_decomposition.png: Understand trend, seasonal and noise components"
echo "- monthly_returns_heatmap.png: View seasonal patterns by year and month"
echo "- volatility_regimes.png: Identify different market regimes"
echo "- sector_correlation_*.png: See how correlations between sectors change over years"
echo ""
echo "======================================================================" 