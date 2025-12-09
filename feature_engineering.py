# ============================================================================
# FEATURE ENGINEERING SCRIPT
# ============================================================================
# Purpose: Create new predictive features from the cleaned data
# Why? Good features are MORE IMPORTANT than complex models!
# Feature engineering is where domain knowledge (business understanding) meets data
# ============================================================================

# STEP 1: IMPORT LIBRARIES
# ------------------------
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("STARTUP INVESTMENT DATASET - FEATURE ENGINEERING")
print("=" * 80)
print()

# STEP 2: LOAD CLEANED DATA
# --------------------------
# Load the data we cleaned in data_cleaning.py
print("Loading cleaned dataset...")
data_path = "cleaned_data.csv"

df = pd.read_csv(data_path)

# Strip whitespace from column names (in case cleaned_data.csv has spaces)
df.columns = df.columns.str.strip()

# Convert date columns back to datetime (CSV saves them as strings)
date_columns = ['founded_at', 'first_funding_at', 'last_funding_at']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

print(f"✓ Loaded {len(df):,} companies")
print()

# STEP 3: TEMPORAL FEATURES (Time-based features)
# ------------------------------------------------
# These features capture how old the company is and funding timing
# Why? Older companies might be more stable, timing matters in VC!
print("-" * 80)
print("CREATING: Temporal (time-based) features")
print("-" * 80)
print()

# FEATURE 1: Company Age (in years)
# ----------------------------------
# How old was the company at the time of data snapshot (around 2015)?
# Hypothesis: Older companies might have higher survival rates
if 'founded_year' in df.columns:
    # Assume data snapshot is from 2015 (based on exploration)
    SNAPSHOT_YEAR = 2015

    # Calculate company age
    df['company_age_years'] = SNAPSHOT_YEAR - df['founded_year']

    # Handle edge cases
    # - Negative age (founded after 2015) → set to 0
    # - Missing founded_year → set to median age
    df['company_age_years'] = df['company_age_years'].apply(lambda x: max(0, x) if pd.notna(x) else np.nan)

    # Fill missing with median
    median_age = df['company_age_years'].median()
    df['company_age_years'] = df['company_age_years'].fillna(median_age)

    print(f"✓ company_age_years created")
    print(f"  Range: {df['company_age_years'].min():.0f} to {df['company_age_years'].max():.0f} years")
    print(f"  Average: {df['company_age_years'].mean():.1f} years")
    print()

# FEATURE 2: Years to First Funding
# ----------------------------------
# How long after founding did the company get first funding?
# Hypothesis: Companies that get funded quickly might be stronger
if 'founded_at' in df.columns and 'first_funding_at' in df.columns:
    # Calculate time difference in days, then convert to years
    time_diff = (df['first_funding_at'] - df['founded_at']).dt.days / 365.25

    # Create feature
    df['years_to_first_funding'] = time_diff

    # Handle edge cases
    # - Negative values (funded before founded?) → set to 0
    # - Very large values (> 50 years) → likely data errors, set to median
    df['years_to_first_funding'] = df['years_to_first_funding'].apply(
        lambda x: max(0, x) if pd.notna(x) and x < 50 else np.nan
    )

    # Fill missing with median
    median_time = df['years_to_first_funding'].median()
    df['years_to_first_funding'] = df['years_to_first_funding'].fillna(median_time)

    print(f"✓ years_to_first_funding created")
    print(f"  Average: {df['years_to_first_funding'].mean():.2f} years")
    print()

# FEATURE 3: Funding Duration
# ----------------------------
# How long did the company receive funding over?
# Hypothesis: Longer funding duration = more sustained investor interest
if 'first_funding_at' in df.columns and 'last_funding_at' in df.columns:
    # Calculate time difference
    time_diff = (df['last_funding_at'] - df['first_funding_at']).dt.days / 365.25

    df['funding_duration_years'] = time_diff

    # Handle missing and negative values
    df['funding_duration_years'] = df['funding_duration_years'].apply(
        lambda x: max(0, x) if pd.notna(x) else 0
    )

    print(f"✓ funding_duration_years created")
    print(f"  Average: {df['funding_duration_years'].mean():.2f} years")
    print()

# STEP 4: FUNDING-BASED FEATURES
# -------------------------------
# These capture patterns in how companies raised money
# Why? Different funding patterns indicate different growth strategies
print("-" * 80)
print("CREATING: Funding-based features")
print("-" * 80)
print()

# FEATURE 4-7: Early Stage Funding Indicators
# --------------------------------------------
# Binary flags: Did the company raise seed/angel/venture funding?
# Hypothesis: Companies with early stage funding are more likely to succeed
funding_types = {
    'has_seed': 'seed',
    'has_angel': 'angel',
    'has_venture': 'venture',
    'has_grant': 'grant'
}

for feature_name, column_name in funding_types.items():
    if column_name in df.columns:
        # Create binary feature: 1 if raised this type, 0 otherwise
        df[feature_name] = (df[column_name] > 0).astype(int)
        count = df[feature_name].sum()
        pct = (count / len(df)) * 100
        print(f"✓ {feature_name:15} created: {count:6,} companies ({pct:5.1f}%) have this")

print()

# FEATURE 8: Total Funding Rounds
# --------------------------------
# Count how many different types of funding the company raised
# Hypothesis: More rounds = more investor validation
round_columns = ['seed', 'angel', 'venture', 'round_A', 'round_B', 'round_C',
                 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']

# Count how many rounds have values > 0
if all(col in df.columns for col in round_columns):
    df['total_funding_rounds'] = sum((df[col] > 0).astype(int) for col in round_columns)

    print(f"✓ total_funding_rounds created")
    print(f"  Average: {df['total_funding_rounds'].mean():.2f} rounds")
    print(f"  Max: {df['total_funding_rounds'].max():.0f} rounds")
    print()
else:
    # Fallback: use the funding_rounds column if it exists
    if 'funding_rounds' in df.columns:
        df['total_funding_rounds'] = df['funding_rounds']
        print(f"✓ total_funding_rounds created (using existing funding_rounds column)")
        print()

# FEATURE 9: Average Funding Per Round
# -------------------------------------
# Total funding divided by number of rounds
# Hypothesis: Larger average rounds = stronger company signal
if 'funding_total_usd' in df.columns and 'total_funding_rounds' in df.columns:
    # Avoid division by zero
    df['avg_funding_per_round'] = df.apply(
        lambda row: row['funding_total_usd'] / row['total_funding_rounds']
        if row['total_funding_rounds'] > 0 else 0,
        axis=1
    )

    print(f"✓ avg_funding_per_round created")
    # Only show stats for companies with funding
    funded = df[df['avg_funding_per_round'] > 0]['avg_funding_per_round']
    print(f"  Average (for funded companies): ${funded.mean():,.0f}")
    print()

# FEATURE 10: Log-transformed Total Funding
# ------------------------------------------
# Why log transform? Funding amounts are heavily skewed (few companies raise billions)
# Log transformation makes the distribution more normal for models
# Formula: log(x + 1) to handle zero values
if 'funding_total_usd' in df.columns:
    df['log_funding_total'] = np.log1p(df['funding_total_usd'])

    print(f"✓ log_funding_total created (log-transformed funding)")
    print(f"  This helps models handle the huge range in funding amounts")
    print()

# STEP 5: GEOGRAPHIC FEATURES
# ----------------------------
# Location matters A LOT in startups!
# Why? Silicon Valley vs elsewhere = different ecosystems
print("-" * 80)
print("CREATING: Geographic features")
print("-" * 80)
print()

# FEATURE 11: Is company in USA?
# -------------------------------
# Hypothesis: US companies might have different success rates
if 'country_code' in df.columns:
    df['is_usa'] = (df['country_code'] == 'USA').astype(int)

    count = df['is_usa'].sum()
    pct = (count / len(df)) * 100
    print(f"✓ is_usa created: {count:,} companies ({pct:.1f}%) in USA")
    print()

# FEATURE 12: Is company in major tech hub?
# ------------------------------------------
# Tech hubs: California, New York, Massachusetts, Texas, Washington
# Hypothesis: Tech hub companies have better access to capital and talent
if 'state_code' in df.columns:
    tech_hub_states = ['CA', 'NY', 'MA', 'TX', 'WA']
    df['is_tech_hub'] = df['state_code'].isin(tech_hub_states).astype(int)

    count = df['is_tech_hub'].sum()
    pct = (count / len(df)) * 100
    print(f"✓ is_tech_hub created: {count:,} companies ({pct:.1f}%) in major tech hubs")
    print()

# STEP 6: CATEGORY/INDUSTRY FEATURES
# -----------------------------------
# What industry is the company in?
# Why? Software companies have different economics than biotech
print("-" * 80)
print("CREATING: Category/industry features")
print("-" * 80)
print()

# FEATURE 13: Category Count
# ---------------------------
# How many categories is the company tagged with?
# Hypothesis: Focused companies (fewer categories) might do better
if 'category_list' in df.columns:
    # Categories are separated by pipes: "|Software|Mobile|"
    # Count by splitting on '|' and removing empty strings
    df['category_count'] = df['category_list'].apply(
        lambda x: len([c for c in str(x).split('|') if c.strip()]) if pd.notna(x) else 0
    )

    print(f"✓ category_count created")
    print(f"  Average: {df['category_count'].mean():.2f} categories per company")
    print()

# FEATURES 14-17: Industry Indicators
# ------------------------------------
# Binary flags for major industries
# These are important because different industries have different success patterns
industry_features = {
    'is_software': 'Software',
    'is_biotech': 'Biotechnology',
    'is_mobile': 'Mobile',
    'is_ecommerce': 'E-Commerce'
}

if 'category_list' in df.columns:
    for feature_name, keyword in industry_features.items():
        # Check if keyword appears in category_list
        df[feature_name] = df['category_list'].str.contains(keyword, case=False, na=False).astype(int)

        count = df[feature_name].sum()
        pct = (count / len(df)) * 100
        print(f"✓ {feature_name:15} created: {count:6,} companies ({pct:5.1f}%)")

    print()

# STEP 7: FUNDING STAGE FEATURES
# -------------------------------
# How far did the company progress in funding rounds?
# Why? Reaching later rounds (C, D, E) indicates proven traction
print("-" * 80)
print("CREATING: Funding stage features")
print("-" * 80)
print()

# FEATURE 18: Maximum Round Reached
# ----------------------------------
# Encode as ordinal: A=1, B=2, C=3, etc.
# Hypothesis: Companies that reach later stages are more mature
round_mapping = {
    'round_A': 1, 'round_B': 2, 'round_C': 3, 'round_D': 4,
    'round_E': 5, 'round_F': 6, 'round_G': 7, 'round_H': 8
}

max_rounds = []
for idx, row in df.iterrows():
    max_round = 0
    for round_col, value in round_mapping.items():
        if round_col in df.columns and row[round_col] > 0:
            max_round = max(max_round, value)
    max_rounds.append(max_round)

df['max_round_reached'] = max_rounds

print(f"✓ max_round_reached created")
print(f"  0 = No institutional rounds, 1 = Series A, 2 = Series B, etc.")
print(f"  Average: {df['max_round_reached'].mean():.2f}")
print()

# STEP 8: HANDLE REMAINING MISSING VALUES
# ----------------------------------------
# Fill any remaining NaN values to ensure models can use all rows
print("-" * 80)
print("HANDLING: Remaining missing values")
print("-" * 80)
print()

# Get all numeric columns we created
feature_columns = [
    'company_age_years', 'years_to_first_funding', 'funding_duration_years',
    'has_seed', 'has_angel', 'has_venture', 'has_grant',
    'total_funding_rounds', 'avg_funding_per_round', 'log_funding_total',
    'is_usa', 'is_tech_hub', 'category_count',
    'is_software', 'is_biotech', 'is_mobile', 'is_ecommerce',
    'max_round_reached'
]

# Check for missing values
for col in feature_columns:
    if col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            # Fill with median for numeric features
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"✓ {col}: Filled {missing:,} missing values with median ({median_val:.2f})")

print()
print("✓ All missing values handled")
print()

# STEP 9: SELECT FINAL FEATURE SET
# ---------------------------------
# Choose which features to keep for modeling
# Why? We want features that are:
# 1. Predictive (correlated with success)
# 2. Available at prediction time (no data leakage)
# 3. Not redundant (avoid multicollinearity)
print("-" * 80)
print("SELECTING: Final feature set for modeling")
print("-" * 80)
print()

# Core features to keep
core_features = ['funding_total_usd', 'log_funding_total', 'funding_rounds']

# Our engineered features
engineered_features = [col for col in feature_columns if col in df.columns]

# Target variables (what we're predicting)
target_features = ['survival_target', 'series_a_target']

# Identifier (to track companies)
identifier = ['permalink', 'name']

# Combine all
features_to_keep = identifier + core_features + engineered_features + target_features

# Keep only these columns
df_features = df[features_to_keep].copy()

print(f"✓ Selected {len(features_to_keep)} columns for modeling:")
print(f"  - {len(identifier)} identifiers (company name/id)")
print(f"  - {len(core_features)} core features")
print(f"  - {len(engineered_features)} engineered features")
print(f"  - {len(target_features)} target variables")
print()

# STEP 10: FEATURE SUMMARY STATISTICS
# ------------------------------------
# Show summary of all features for review
print("-" * 80)
print("FEATURE SUMMARY STATISTICS")
print("-" * 80)
print()

# Get numeric features only
numeric_features = core_features + engineered_features

print("Numeric features (showing first 10):")
print()
print(df_features[numeric_features[:10]].describe().round(2))
print()

# STEP 11: SAVE FEATURE-ENGINEERED DATA
# --------------------------------------
# Save to CSV for use in model building scripts
print("-" * 80)
print("SAVING: Feature-engineered data")
print("-" * 80)
print()

output_path = "features_data.csv"
df_features.to_csv(output_path, index=False, encoding='utf-8')

print(f"✓ Saved feature-engineered data to: {output_path}")
print(f"  Rows: {len(df_features):,}")
print(f"  Columns: {df_features.shape[1]}")
print()

# STEP 12: FEATURE ENGINEERING SUMMARY
# -------------------------------------
print("=" * 80)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 80)
print()

print("Features created by category:")
print()
print("1. TEMPORAL FEATURES (3):")
print("   - company_age_years: How old is the company?")
print("   - years_to_first_funding: How quickly did they get funded?")
print("   - funding_duration_years: How long did funding last?")
print()

print("2. FUNDING FEATURES (7):")
print("   - has_seed, has_angel, has_venture, has_grant: Early funding types")
print("   - total_funding_rounds: How many rounds?")
print("   - avg_funding_per_round: Average round size")
print("   - log_funding_total: Log-transformed total funding")
print()

print("3. GEOGRAPHIC FEATURES (2):")
print("   - is_usa: Located in USA?")
print("   - is_tech_hub: Located in major tech hub (CA, NY, MA, TX, WA)?")
print()

print("4. INDUSTRY FEATURES (5):")
print("   - category_count: How many categories?")
print("   - is_software, is_biotech, is_mobile, is_ecommerce: Industry flags")
print()

print("5. FUNDING STAGE FEATURES (1):")
print("   - max_round_reached: Furthest funding round (A=1, B=2, etc.)")
print()

print("Total engineered features: 18")
print()

print("Why these features matter for predicting startup success:")
print("  ✓ Temporal: Timing and age indicate maturity")
print("  ✓ Funding: Capital and investor validation signal strength")
print("  ✓ Geographic: Location affects ecosystem access")
print("  ✓ Industry: Different sectors have different dynamics")
print("  ✓ Stage: Later stages indicate proven traction")
print()

print("=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print()
print("Next script to run: model_survival.py")
print()
