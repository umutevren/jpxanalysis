import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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


def load_stratified_sample(filepath, strat_column, date_column='Date', sample_size=500000):
    """
    Load a sample of data stratified by a specific column (e.g., sector, market segment)
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    strat_column : str
        Column to stratify by (e.g., 'SecuritiesCode', '33SectorName')
    date_column : str
        Name of the date column
    sample_size : int
        Total number of rows to sample
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with samples proportionally distributed across strata
    """
    print(f"Performing stratified sampling from {filepath} by {strat_column}...")
    
    # Read the file
    full_data = pd.read_csv(filepath)
    
    # Convert date column to datetime
    if date_column in full_data.columns:
        full_data[date_column] = pd.to_datetime(full_data[date_column])
    
    # Get value counts of stratification column
    if strat_column not in full_data.columns:
        print(f"Warning: {strat_column} not found in data. Using random sampling instead.")
        return full_data.sample(min(sample_size, len(full_data)), random_state=42)
    
    strat_counts = full_data[strat_column].value_counts(normalize=True)
    
    # Calculate how many samples to take from each stratum
    sample_counts = (strat_counts * sample_size).astype(int).to_dict()
    
    # Adjust to ensure we get exactly sample_size rows (due to rounding)
    total = sum(sample_counts.values())
    if total < sample_size:
        # Add the remaining samples to the largest stratum
        largest_stratum = strat_counts.index[0]
        sample_counts[largest_stratum] += sample_size - total
    
    # Sample from each stratum
    sampled_data = []
    for stratum, count in sample_counts.items():
        stratum_data = full_data[full_data[strat_column] == stratum]
        # If there are fewer rows than requested, take all rows
        if len(stratum_data) <= count:
            sampled_data.append(stratum_data)
        else:
            sampled_data.append(stratum_data.sample(count, random_state=42))
    
    # Combine all strata
    combined_sample = pd.concat(sampled_data, ignore_index=True)
    print(f"Stratified sampling complete. Sample shape: {combined_sample.shape}")
    
    return combined_sample


def load_hybrid_sample(filepath, date_column='Date', strat_column=None, sample_size=500000, chunks=5):
    """
    Load a sample that is both time-aware and stratified
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    date_column : str
        Name of the date column
    strat_column : str or None
        Column to stratify by within each time chunk
    sample_size : int
        Total number of rows to sample
    chunks : int
        Number of time periods to divide the data into
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with samples balanced across time and strata
    """
    print(f"Performing hybrid sampling from {filepath}...")
    
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
        
        # Read full data (this is inefficient but necessary to get strat_column)
        full_data = pd.read_csv(filepath)
        full_data[date_column] = pd.to_datetime(full_data[date_column])
        
        # Filter to current time chunk
        chunk_data = full_data[(full_data[date_column] >= chunk_start) & 
                              (full_data[date_column] < chunk_end)]
        
        # If stratification column is provided, sample within strata in this chunk
        if strat_column and strat_column in full_data.columns:
            strat_counts = chunk_data[strat_column].value_counts(normalize=True)
            
            # Calculate samples per stratum
            stratum_samples = (strat_counts * rows_per_chunk).astype(int).to_dict()
            
            # Adjust to ensure we get exactly rows_per_chunk rows
            total = sum(stratum_samples.values())
            if total < rows_per_chunk and len(strat_counts) > 0:
                largest_stratum = strat_counts.index[0]
                stratum_samples[largest_stratum] += rows_per_chunk - total
            
            # Sample from each stratum in this time chunk
            chunk_samples = []
            for stratum, count in stratum_samples.items():
                stratum_data = chunk_data[chunk_data[strat_column] == stratum]
                if len(stratum_data) <= count:
                    chunk_samples.append(stratum_data)
                else:
                    chunk_samples.append(stratum_data.sample(count, random_state=42+i))
            
            # Combine strata for this chunk
            if chunk_samples:
                chunk_sample = pd.concat(chunk_samples, ignore_index=True)
                sampled_data.append(chunk_sample)
        else:
            # Just do random sampling in this chunk
            if len(chunk_data) > rows_per_chunk:
                chunk_sample = chunk_data.sample(rows_per_chunk, random_state=42+i)
            else:
                chunk_sample = chunk_data  # Take all data if less than requested
                
            sampled_data.append(chunk_sample)
    
    # Combine all chunks
    combined_sample = pd.concat(sampled_data, ignore_index=True)
    print(f"Hybrid sampling complete. Sample shape: {combined_sample.shape}")
    
    return combined_sample 