# ============================================================================
# DATA EXPLORATION SCRIPT
# ============================================================================
# Purpose: Understand the startup investment dataset before building models
# This is ALWAYS the first step in any data science project
# Why? You need to know what data you have before you can use it!
# ============================================================================

# STEP 1: IMPORT LIBRARIES
# ------------------------
# pandas: The main library for working with tabular data (like Excel/CSV)
# It lets us load, view, and manipulate data easily
import pandas as pd

# numpy: Library for numerical operations (math, statistics)
import numpy as np

print("=" * 80)
print("STARTUP INVESTMENT DATASET - INITIAL EXPLORATION")
print("=" * 80)
print()

# STEP 2: LOAD THE DATASET
# -------------------------
# We're loading the CSV file that was downloaded by ds.py
# encoding='latin-1': This handles special characters in company names
# Why not UTF-8? The dataset was created with latin-1 encoding
print("Loading dataset...")
print()

# The file path - this is where ds.py downloaded the data
data_path = "investments_VC.csv"

# Load the CSV file into a pandas DataFrame (think of it like an Excel spreadsheet)
df = pd.read_csv(data_path, encoding='latin-1')

# IMPORTANT: Strip whitespace from column names
# Some columns have spaces like ' funding_total_usd ' instead of 'funding_total_usd'
df.columns = df.columns.str.strip()

print(f"✓ Dataset loaded successfully!")
print()

# STEP 3: BASIC DATASET INFORMATION
# ----------------------------------
# Let's see what we're working with
print("-" * 80)
print("BASIC DATASET INFO")
print("-" * 80)
print()

# Shape tells us (number of rows, number of columns)
# Rows = individual startup companies
# Columns = features/attributes about each company
print(f"Dataset Shape: {df.shape[0]:,} rows (companies) x {df.shape[1]} columns (features)")
print()

# Total number of data points
total_cells = df.shape[0] * df.shape[1]
print(f"Total data points: {total_cells:,}")
print()

# STEP 4: COLUMN NAMES AND DATA TYPES
# ------------------------------------
# Let's see all the columns we have available
print("-" * 80)
print("COLUMN NAMES AND DATA TYPES")
print("-" * 80)
print()

# .info() gives us column names, data types, and non-null counts
# Non-null = how many values are NOT missing
print(df.info())
print()

# STEP 5: PREVIEW THE DATA
# -------------------------
# Let's look at the first few rows to understand the data structure
print("-" * 80)
print("FIRST 5 ROWS (Sample Data)")
print("-" * 80)
print()
print(df.head())
print()

# STEP 6: MISSING VALUES ANALYSIS
# --------------------------------
# Missing data is VERY common in real-world datasets
# We need to know what's missing to decide how to handle it
print("-" * 80)
print("MISSING VALUES ANALYSIS")
print("-" * 80)
print()

# Calculate missing values for each column
missing_values = df.isnull().sum()

# Calculate percentage missing
missing_percentage = (missing_values / len(df)) * 100

# Create a summary DataFrame
missing_summary = pd.DataFrame({
    'Column': missing_values.index,
    'Missing_Count': missing_values.values,
    'Missing_Percentage': missing_percentage.values
})

# Sort by missing count (highest first)
missing_summary = missing_summary.sort_values('Missing_Count', ascending=False)

# Only show columns that have missing values
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]

print(f"Columns with missing values: {len(missing_summary)} out of {df.shape[1]}")
print()
print(missing_summary.to_string(index=False))
print()

# STEP 7: TARGET VARIABLE #1 - STARTUP STATUS
# --------------------------------------------
# This is what we want to predict: Will the startup succeed or fail?
# Status can be: 'operating', 'acquired', 'closed', or missing
print("-" * 80)
print("TARGET VARIABLE #1: STARTUP STATUS (Success/Failure)")
print("-" * 80)
print()

# Count how many companies have each status
status_counts = df['status'].value_counts()
status_percentage = (df['status'].value_counts(normalize=True) * 100).round(2)

print("Distribution of company status:")
print()
for status, count in status_counts.items():
    pct = status_percentage[status]
    print(f"  {status:15} : {count:6,} companies ({pct:5.2f}%)")

# How many have missing status?
missing_status = df['status'].isnull().sum()
if missing_status > 0:
    missing_pct = (missing_status / len(df)) * 100
    print(f"  {'Unknown':15} : {missing_status:6,} companies ({missing_pct:5.2f}%)")

print()

# For our model, we'll classify:
# - Success = 'operating' or 'acquired' (company is still alive or successfully exited)
# - Failure = 'closed' (company shut down)
success_count = status_counts.get('operating', 0) + status_counts.get('acquired', 0)
failure_count = status_counts.get('closed', 0)
total_labeled = success_count + failure_count

print("For survival prediction model:")
print(f"  Success (operating/acquired): {success_count:6,} companies ({success_count/total_labeled*100:5.2f}%)")
print(f"  Failure (closed)           : {failure_count:6,} companies ({failure_count/total_labeled*100:5.2f}%)")
print()

# STEP 8: TARGET VARIABLE #2 - SERIES A FUNDING
# ----------------------------------------------
# This is our second prediction: Will the startup raise Series A?
# Series A is the first major institutional funding round
print("-" * 80)
print("TARGET VARIABLE #2: SERIES A FUNDING")
print("-" * 80)
print()

# Check if the round_A column exists and has data
# We need to handle the fact that round_A might have text like " -  " or actual numbers
if 'round_A' in df.columns:
    # Try to convert round_A to numeric (will convert non-numbers to NaN)
    round_a_numeric = pd.to_numeric(df['round_A'], errors='coerce')

    # Count how many have Series A funding (> 0)
    has_series_a = (round_a_numeric > 0).sum()
    no_series_a = (round_a_numeric == 0).sum() + round_a_numeric.isnull().sum()

    total_for_series_a = has_series_a + no_series_a

    print("Distribution of Series A funding:")
    print(f"  Has Series A funding    : {has_series_a:6,} companies ({has_series_a/total_for_series_a*100:5.2f}%)")
    print(f"  No Series A funding     : {no_series_a:6,} companies ({no_series_a/total_for_series_a*100:5.2f}%)")
    print()
else:
    print("Warning: 'round_A' column not found in dataset!")
    print()

# STEP 9: FUNDING STATISTICS
# ---------------------------
# Let's understand the funding amounts in this dataset
print("-" * 80)
print("FUNDING STATISTICS")
print("-" * 80)
print()

# The funding_total_usd column has the total funding raised
# But it's stored as text with commas and spaces, so we need to clean it first
if 'funding_total_usd' in df.columns:
    # Remove commas and spaces, then convert to numeric
    funding_cleaned = df['funding_total_usd'].astype(str).str.replace(',', '').str.replace(' ', '')
    funding_numeric = pd.to_numeric(funding_cleaned, errors='coerce')

    # Calculate statistics only for companies with funding data
    funded_companies = funding_numeric[funding_numeric > 0]

    print(f"Companies with funding data: {len(funded_companies):,}")
    print()
    print("Funding amount statistics (in USD):")
    print(f"  Minimum    : ${funded_companies.min():>15,.0f}")
    print(f"  25th %ile  : ${funded_companies.quantile(0.25):>15,.0f}")
    print(f"  Median     : ${funded_companies.quantile(0.50):>15,.0f}")
    print(f"  Mean       : ${funded_companies.mean():>15,.0f}")
    print(f"  75th %ile  : ${funded_companies.quantile(0.75):>15,.0f}")
    print(f"  Maximum    : ${funded_companies.max():>15,.0f}")
    print()

    # Show funding by magnitude
    print("Funding ranges:")
    print(f"  Under $1M      : {(funded_companies < 1_000_000).sum():6,} companies")
    print(f"  $1M - $10M     : {((funded_companies >= 1_000_000) & (funded_companies < 10_000_000)).sum():6,} companies")
    print(f"  $10M - $50M    : {((funded_companies >= 10_000_000) & (funded_companies < 50_000_000)).sum():6,} companies")
    print(f"  $50M - $100M   : {((funded_companies >= 50_000_000) & (funded_companies < 100_000_000)).sum():6,} companies")
    print(f"  Over $100M     : {(funded_companies >= 100_000_000).sum():6,} companies")
    print()

# STEP 10: GEOGRAPHIC DISTRIBUTION
# ---------------------------------
# Where are these startups located?
print("-" * 80)
print("GEOGRAPHIC DISTRIBUTION")
print("-" * 80)
print()

if 'country_code' in df.columns:
    top_countries = df['country_code'].value_counts().head(10)
    print("Top 10 countries by number of startups:")
    print()
    for country, count in top_countries.items():
        pct = (count / len(df)) * 100
        print(f"  {country:5} : {count:6,} companies ({pct:5.2f}%)")
    print()

if 'state_code' in df.columns:
    # Filter to only US states
    us_companies = df[df['country_code'] == 'USA']
    if len(us_companies) > 0:
        top_states = us_companies['state_code'].value_counts().head(10)
        print("Top 10 US states by number of startups:")
        print()
        for state, count in top_states.items():
            pct = (count / len(us_companies)) * 100
            print(f"  {state:5} : {count:6,} companies ({pct:5.2f}%)")
        print()

# STEP 11: INDUSTRY/CATEGORY DISTRIBUTION
# ----------------------------------------
# What industries do these startups operate in?
print("-" * 80)
print("INDUSTRY/CATEGORY INSIGHTS")
print("-" * 80)
print()

if 'market' in df.columns:
    top_markets = df['market'].value_counts().head(10)
    print("Top 10 markets/industries:")
    print()
    for market, count in top_markets.items():
        pct = (count / len(df)) * 100
        print(f"  {market:30} : {count:5,} companies ({pct:5.2f}%)")
    print()

# STEP 12: SUMMARY AND KEY INSIGHTS
# ----------------------------------
print("=" * 80)
print("KEY INSIGHTS FROM EXPLORATION")
print("=" * 80)
print()

print("1. Dataset Size:")
print(f"   - We have {df.shape[0]:,} startup companies with {df.shape[1]} features each")
print()

print("2. Data Quality:")
print(f"   - {len(missing_summary)} columns have missing values")
print(f"   - This is normal for real-world data!")
print()

print("3. Prediction Targets:")
print(f"   - Survival: {success_count:,} successes vs {failure_count:,} failures")
print(f"   - Class imbalance: {success_count/failure_count:.1f}:1 ratio (more successes)")
if 'round_A' in df.columns:
    print(f"   - Series A: {has_series_a:,} raised vs {no_series_a:,} didn't raise")
    print(f"   - Class imbalance: {no_series_a/has_series_a:.1f}:1 ratio (fewer raise Series A)")
print()

print("4. Next Steps:")
print("   - Clean the funding amounts (remove commas, convert to numbers)")
print("   - Handle missing values strategically")
print("   - Engineer features from dates, categories, and funding rounds")
print("   - Build models with class imbalance handling")
print()

print("=" * 80)
print("EXPLORATION COMPLETE!")
print("=" * 80)
print()
print("Next script to run: data_cleaning.py")
print()
