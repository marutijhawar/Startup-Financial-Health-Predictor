# ============================================================================
# SERIES A FUNDING PREDICTION MODEL
# ============================================================================
# Purpose: Predict whether a startup will raise Series A funding
# Why? Series A is a critical milestone - this helps identify which companies will scale
# Models: Logistic Regression & Decision Tree (simple, interpretable)
# ============================================================================

# STEP 1: IMPORT LIBRARIES
# ------------------------
import pandas as pd
import numpy as np

# Scikit-learn: Machine learning library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("=" * 80)
print("SERIES A FUNDING PREDICTION MODEL")
print("=" * 80)
print()
print("Goal: Predict if a startup will raise Series A funding (1) or not (0)")
print("Series A = First major institutional funding round")
print()

# STEP 2: LOAD FEATURE-ENGINEERED DATA
# -------------------------------------
print("-" * 80)
print("LOADING DATA")
print("-" * 80)
print()

data_path = "features_data.csv"
df = pd.read_csv(data_path)

print(f"✓ Loaded {len(df):,} companies with {df.shape[1]} features")
print()

# STEP 3: FILTER TO RELEVANT COMPANIES
# -------------------------------------
# Important decision: Only include companies that have raised SOME funding
# Why? Predicting Series A for completely unfunded companies isn't meaningful
# We want to predict: "Given a company raised seed/angel, will they raise Series A?"
print("-" * 80)
print("FILTERING: To funded companies only")
print("-" * 80)
print()

# Filter to companies with at least some funding
# This could be seed, angel, or venture funding
before_filter = len(df)

# Companies with either seed OR angel OR venture OR any funding > 0
funded_mask = (
    (df.get('has_seed', 0) == 1) |
    (df.get('has_angel', 0) == 1) |
    (df.get('has_venture', 0) == 1) |
    (df.get('funding_total_usd', 0) > 0)
)

df_funded = df[funded_mask].copy()

after_filter = len(df_funded)
removed = before_filter - after_filter

print(f"Before filtering: {before_filter:,} companies")
print(f"After filtering:  {after_filter:,} companies (have raised some funding)")
print(f"Removed:          {removed:,} companies (unfunded)")
print()
print("Why filter?")
print("  → We're predicting Series A for startups that already have early funding")
print("  → This makes the prediction more practical for VC use cases")
print()

# STEP 4: PREPARE DATA FOR MODELING
# ----------------------------------
print("-" * 80)
print("PREPARING DATA FOR MODELING")
print("-" * 80)
print()

# Use the same features as survival model for consistency
feature_columns = [
    # Funding features (excluding Series A itself!)
    'funding_total_usd', 'log_funding_total', 'funding_rounds',
    'total_funding_rounds', 'avg_funding_per_round',
    'has_seed', 'has_angel', 'has_venture', 'has_grant',
    # Temporal features
    'company_age_years', 'years_to_first_funding', 'funding_duration_years',
    # Geographic features
    'is_usa', 'is_tech_hub',
    # Industry features
    'category_count', 'is_software', 'is_biotech', 'is_mobile', 'is_ecommerce',
    # Stage features
    'max_round_reached'
]

# Keep only features that exist
feature_columns = [col for col in feature_columns if col in df_funded.columns]

# X = features (inputs)
X = df_funded[feature_columns].copy()

# y = target (what we predict)
# 1 = raised Series A, 0 = did not raise Series A
y = df_funded['series_a_target'].copy()

# Remove any rows where target is missing
valid_indices = y.notna()
X = X[valid_indices]
y = y[valid_indices]

print(f"Features (X): {X.shape[0]:,} companies × {X.shape[1]} features")
print(f"Target (y):   {len(y):,} labels")
print()

# Handle missing values
missing_counts = X.isnull().sum().sum()
if missing_counts > 0:
    print(f"Filling {missing_counts} missing values with median...")
    X = X.fillna(X.median())
    print("✓ Done")
    print()

# STEP 5: CHECK CLASS DISTRIBUTION
# ---------------------------------
# Series A is a significant milestone - we expect class imbalance
# (Most companies DON'T raise Series A)
print("-" * 80)
print("TARGET VARIABLE DISTRIBUTION")
print("-" * 80)
print()

class_counts = y.value_counts()
has_series_a = class_counts.get(1, 0)
no_series_a = class_counts.get(0, 0)
total = has_series_a + no_series_a

print(f"Raised Series A (1):     {has_series_a:6,} companies ({has_series_a/total*100:5.2f}%)")
print(f"Did not raise Series A (0): {no_series_a:6,} companies ({no_series_a/total*100:5.2f}%)")
print()

# Calculate imbalance ratio
if has_series_a > 0:
    imbalance_ratio = no_series_a / has_series_a
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1 (no Series A : Series A)")
    print()
    print("⚠ Significant class imbalance!")
    print("  → Only ~{:.1f}% raise Series A".format(has_series_a/total*100))
    print("  → Will use class_weight='balanced' to handle this")
    print()

# STEP 6: SPLIT DATA INTO TRAIN AND TEST SETS
# --------------------------------------------
print("-" * 80)
print("SPLITTING DATA: Train/Test")
print("-" * 80)
print()

# 80/20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class ratio
)

print(f"Training set:   {len(X_train):6,} companies ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set:       {len(X_test):6,} companies ({len(X_test)/len(X)*100:.1f}%)")
print()

# STEP 7: BUILD MODEL #1 - LOGISTIC REGRESSION
# ---------------------------------------------
print("=" * 80)
print("MODEL #1: LOGISTIC REGRESSION")
print("=" * 80)
print()

print("Predicting: Probability of raising Series A funding")
print("Model: Logistic Regression with balanced class weights")
print()

# Create model
logistic_model = LogisticRegression(
    class_weight='balanced',  # Critical for imbalanced data!
    max_iter=1000,
    random_state=42
)

print("Training Logistic Regression model...")
logistic_model.fit(X_train, y_train)
print("✓ Training complete!")
print()

# STEP 8: EVALUATE LOGISTIC REGRESSION
# -------------------------------------
print("-" * 80)
print("EVALUATING: Logistic Regression")
print("-" * 80)
print()

# Predictions
y_pred_logistic = logistic_model.predict(X_test)
y_pred_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]

# METRIC 1: Accuracy
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Accuracy: {accuracy_logistic:.4f} ({accuracy_logistic*100:.2f}%)")
print()

# METRIC 2: Confusion Matrix
cm_logistic = confusion_matrix(y_test, y_pred_logistic)

print("Confusion Matrix:")
print()
print("                    Predicted")
print("                    No A | Has A")
print("           -----------------------")
print(f"Actual No A    |  {cm_logistic[0][0]:4} | {cm_logistic[0][1]:4}  (TN | FP)")
print(f"       Has A   |  {cm_logistic[1][0]:4} | {cm_logistic[1][1]:4}  (FN | TP)")
print()

tn, fp, fn, tp = cm_logistic.ravel()
print("In VC terms:")
print(f"  True Negatives (TN):  {tn:4} - Correctly predicted won't raise A")
print(f"  False Positives (FP): {fp:4} - Predicted would raise A, but didn't")
print(f"  False Negatives (FN): {fn:4} - Predicted wouldn't raise A, but did!")
print(f"  True Positives (TP):  {tp:4} - Correctly predicted will raise A")
print()

# Which error is worse for a VC?
# False Negative = Missing a good opportunity (could be costly!)
# False Positive = Betting on wrong company (lose money)
print("VC perspective:")
print(f"  False Negatives (FN): {fn:4} - Missed opportunities!")
print(f"  False Positives (FP): {fp:4} - Bad bets")
print()

# METRIC 3: Classification Report
print("Classification Report:")
print()
print(classification_report(y_test, y_pred_logistic, target_names=['No Series A', 'Has Series A']))

print("Key metrics for imbalanced data:")
print("  Precision (Has A): Of predicted 'will raise A', how many actually did?")
print("  Recall (Has A): Of actual Series A raises, how many did we catch?")
print("  F1-score: Balance between precision and recall")
print()

# STEP 9: FEATURE IMPORTANCE (LOGISTIC REGRESSION)
# -------------------------------------------------
print("-" * 80)
print("FEATURE IMPORTANCE: What predicts Series A success?")
print("-" * 80)
print()

coefficients = logistic_model.coef_[0]

feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)

print("Top 10 most important features for Series A:")
print()
print(f"{'Feature':<30} {'Coefficient':>12} {'Impact':>10}")
print("-" * 55)
for idx, row in feature_importance_df.head(10).iterrows():
    impact = "Increases" if row['Coefficient'] > 0 else "Decreases"
    print(f"{row['Feature']:<30} {row['Coefficient']:>12.4f} {impact:>10}")

print()
print("What this tells VCs:")
print("  Positive → Feature increases likelihood of raising Series A")
print("  Negative → Feature decreases likelihood of raising Series A")
print()

# STEP 10: BUILD MODEL #2 - DECISION TREE
# ----------------------------------------
print("=" * 80)
print("MODEL #2: DECISION TREE")
print("=" * 80)
print()

print("Decision Tree: Clear rules for Series A prediction")
print()

# Create model
tree_model = DecisionTreeClassifier(
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

print("Training Decision Tree model...")
tree_model.fit(X_train, y_train)
print("✓ Training complete!")
print()

# STEP 11: EVALUATE DECISION TREE
# --------------------------------
print("-" * 80)
print("EVALUATING: Decision Tree")
print("-" * 80)
print()

y_pred_tree = tree_model.predict(X_test)

# Accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Accuracy: {accuracy_tree:.4f} ({accuracy_tree*100:.2f}%)")
print()

# Confusion Matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)

print("Confusion Matrix:")
print()
print("                    Predicted")
print("                    No A | Has A")
print("           -----------------------")
print(f"Actual No A    |  {cm_tree[0][0]:4} | {cm_tree[0][1]:4}  (TN | FP)")
print(f"       Has A   |  {cm_tree[1][0]:4} | {cm_tree[1][1]:4}  (FN | TP)")
print()

# Classification Report
print("Classification Report:")
print()
print(classification_report(y_test, y_pred_tree, target_names=['No Series A', 'Has Series A']))

# STEP 12: FEATURE IMPORTANCE (DECISION TREE)
# --------------------------------------------
print("-" * 80)
print("FEATURE IMPORTANCE: Decision Tree")
print("-" * 80)
print()

importances = tree_model.feature_importances_

feature_importance_tree = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
})

feature_importance_tree = feature_importance_tree.sort_values('Importance', ascending=False)

print("Top 10 most important features:")
print()
print(f"{'Feature':<30} {'Importance':>12} {'%':>8}")
print("-" * 52)
for idx, row in feature_importance_tree.head(10).iterrows():
    pct = row['Importance'] * 100
    print(f"{row['Feature']:<30} {row['Importance']:>12.4f} {pct:>7.2f}%")

print()

# STEP 13: MODEL COMPARISON
# --------------------------
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print()

print(f"{'Model':<25} {'Accuracy':>10} {'Notes':<35}")
print("-" * 75)
print(f"{'Logistic Regression':<25} {accuracy_logistic:>10.4f} {'Linear relationships, probabilities':<35}")
print(f"{'Decision Tree':<25} {accuracy_tree:>10.4f} {'Clear decision rules':<35}")
print()

if accuracy_logistic > accuracy_tree:
    print(f"✓ Logistic Regression performs better ({accuracy_logistic:.4f} vs {accuracy_tree:.4f})")
    best_model = "Logistic Regression"
elif accuracy_tree > accuracy_logistic:
    print(f"✓ Decision Tree performs better ({accuracy_tree:.4f} vs {accuracy_logistic:.4f})")
    best_model = "Decision Tree"
else:
    print("✓ Both models perform equally")
    best_model = "Both"

print()

# STEP 14: BUSINESS INSIGHTS FOR VCS
# -----------------------------------
print("=" * 80)
print("BUSINESS INSIGHTS: What drives Series A success?")
print("=" * 80)
print()

# Top factors that INCREASE Series A likelihood
top_positive = feature_importance_df[feature_importance_df['Coefficient'] > 0].head(5)
print("✓ Factors that INCREASE Series A likelihood:")
for idx, row in top_positive.iterrows():
    print(f"  • {row['Feature']}")
print()

# Top factors that DECREASE Series A likelihood
top_negative = feature_importance_df[feature_importance_df['Coefficient'] < 0].head(5)
print("⚠ Factors that DECREASE Series A likelihood:")
for idx, row in top_negative.iterrows():
    print(f"  • {row['Feature']}")
print()

# STEP 15: PRACTICAL VC APPLICATIONS
# -----------------------------------
print("-" * 80)
print("PRACTICAL APPLICATIONS FOR PE/VC")
print("-" * 80)
print()

print("How to use this model in practice:")
print()
print("1. PIPELINE PRIORITIZATION")
print("   → Score incoming deals on Series A potential")
print("   → Focus on high-probability companies")
print()

print("2. DUE DILIGENCE")
print("   → Identify red flags early")
print("   → Understand which metrics matter most")
print()

print("3. PORTFOLIO MONITORING")
print("   → Track portfolio companies' Series A readiness")
print("   → Provide targeted support based on weak signals")
print()

print("4. MARKET INSIGHTS")
print("   → Understand what separates winners from losers")
print("   → Benchmark against successful patterns")
print()

# STEP 16: MODEL PERFORMANCE SUMMARY
# -----------------------------------
print("=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print()

print(f"Dataset: {len(X):,} funded startups")
print(f"Target: {has_series_a:,} ({has_series_a/total*100:.1f}%) raised Series A")
print()
print(f"Best Model: {best_model}")
print(f"Best Accuracy: {max(accuracy_logistic, accuracy_tree)*100:.2f}%")
print()

# Calculate baseline (what if we just predicted "no Series A" for everyone?)
baseline_accuracy = no_series_a / total
improvement = max(accuracy_logistic, accuracy_tree) / baseline_accuracy

print(f"Baseline (always predict 'No'): {baseline_accuracy*100:.2f}%")
print(f"Our model improvement: {improvement:.2f}x better than baseline")
print()

print("Model is ready for:")
print("  ✓ Deal flow scoring")
print("  ✓ Investment decision support")
print("  ✓ Portfolio risk assessment")
print("  ✓ Pattern discovery in successful startups")
print()

print("=" * 80)
print("SERIES A PREDICTION MODEL COMPLETE!")
print("=" * 80)
print()
print("You now have two models:")
print("  1. model_survival.py → Predicts startup success/failure")
print("  2. model_series_a.py → Predicts Series A funding")
print()
print("Both models are simple, interpretable, and ready for your portfolio!")
print()
