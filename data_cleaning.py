# ============================================================================
# DATA CLEANING SCRIPT
# ============================================================================
# Purpose: Clean and prepare the raw startup investment data for modeling
# Why? Raw data is messy - we need to fix it before machine learning models can use it
# This is 70% of the work in data science!
# ============================================================================

# STEP 1: IMPORT LIBRARIES
# ------------------------
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("STARTUP INVESTMENT DATASET - DATA CLEANING")
print("=" * 80)

print("STEP 2: LOAD THE RAW DATASET")
# -----------------------------
data_path = "investments_VC.csv"
df = pd.read_csv(data_path, encoding='latin-1')
df.columns = df.columns.str.strip()

print(f"✓ Loaded {len(df):,} companies with {df.shape[1]} columns")

# Keep a copy of original shape to track our cleaning progress
original_rows = len(df)

# STEP 3: CLEAN FUNDING_TOTAL_USD COLUMN
# ---------------------------------------
df['funding_total_usd'] = df['funding_total_usd'].astype(str).str.replace(',', '').str.replace(' ', '')
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
df['funding_total_usd'] = df['funding_total_usd'].fillna(0)

print(f"✓ Cleaned funding_total_usd (Range: ${df['funding_total_usd'].min():,.0f} - ${df['funding_total_usd'].max():,.0f})")

print("STEP 4: CLEAN ALL FUNDING ROUND COLUMNS")
# ----------------------------------------
funding_columns = [
    'seed', 'angel', 'venture', 'round_A', 'round_B', 'round_C', 'round_D',
    'round_E', 'round_F', 'round_G', 'round_H', 'funding_rounds',
    'convertible_note', 'debt_financing', 'equity_crowdfunding',
    'grant', 'post_ipo_debt', 'post_ipo_equity', 'private_equity',
    'product_crowdfunding', 'secondary_market', 'undisclosed'
]

cleaned_count = 0
for col in funding_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)
        cleaned_count += 1

print(f"✓ Cleaned {cleaned_count} funding round columns")

print("STEP 5: CLEAN DATE COLUMNS")
# ---------------------------
date_columns = ['founded_at', 'first_funding_at', 'last_funding_at']

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

print("✓ Converted date columns to datetime format")

print("STEP 6: HANDLE FOUNDED_YEAR AND FOUNDED_MONTH")
# ----------------------------------------------
if 'founded_year' in df.columns:
    df['founded_year'] = pd.to_numeric(df['founded_year'], errors='coerce')

if 'founded_month' in df.columns:
    df['founded_month'] = pd.to_numeric(df['founded_month'], errors='coerce')

print("✓ Cleaned founded_year and founded_month")

print("STEP 7: CREATE TARGET VARIABLE #1 - SURVIVAL")
# ---------------------------------------------
df['survival_target'] = df['status'].apply(
    lambda x: 1 if x in ['operating', 'acquired'] else (0 if x == 'closed' else np.nan)
)

survival_counts = df['survival_target'].value_counts()
print(f"✓ Created survival_target: {survival_counts.get(1, 0):,} success, {survival_counts.get(0, 0):,} failure")

print("STEP 8: CREATE TARGET VARIABLE #2 - SERIES A")
# ---------------------------------------------
if 'round_A' in df.columns:
    df['series_a_target'] = (df['round_A'] > 0).astype(int)
    series_a_counts = df['series_a_target'].value_counts()
    print(f"✓ Created series_a_target: {series_a_counts.get(1, 0):,} raised, {series_a_counts.get(0, 0):,} didn't raise")

print("STEP 9: REMOVE COMPANIES WITH UNKNOWN STATUS")
# ---------------------------------------------
before_removal = len(df)
df = df[df['survival_target'].notna()]
after_removal = len(df)

print(f"✓ Removed {before_removal - after_removal:,} companies with unknown status ({after_removal:,} remaining)")

print("STEP 10: HANDLE MISSING VALUES IN KEY COLUMNS")
# ----------------------------------------------
categorical_columns = ['country_code', 'state_code', 'region', 'city', 'market']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

if 'category_list' in df.columns:
    df['category_list'] = df['category_list'].fillna('')

print("✓ Handled missing values in categorical columns")

print("STEP 11: DATA VALIDATION")
# -------------------------
negative_funding = (df['funding_total_usd'] < 0).sum()
if negative_funding > 0:
    df.loc[df['funding_total_usd'] < 0, 'funding_total_usd'] = 0

print("✓ Data validation complete")

print("STEP 12: SAVE CLEANED DATA")
# ---------------------------
output_path = "cleaned_data.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"✓ Saved cleaned data: {len(df):,} rows, {df.shape[1]} columns")

print("STEP 13: SUMMARY")
# ----------------
print("=" * 80)
print(f"CLEANING COMPLETE: {original_rows:,} → {len(df):,} companies")
print("Next: feature_engineering.py")
print("=" * 80)
