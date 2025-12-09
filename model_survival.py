# ============================================================================
# STARTUP SURVIVAL PREDICTION MODEL
# ============================================================================
# Purpose: Predict whether a startup will succeed (operating/acquired) or fail (closed)
# Why? This helps VCs/PEs identify risk factors and success patterns
# Models: Logistic Regression & Decision Tree (simple, interpretable)
# ============================================================================

# STEP 1: IMPORT LIBRARIES
# ------------------------
import pandas as pd
import numpy as np

# Scikit-learn: The main machine learning library in Python
# It has tools for models, evaluation, and data splitting
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.linear_model import LogisticRegression    # Logistic Regression model
from sklearn.tree import DecisionTreeClassifier        # Decision Tree model
from sklearn.metrics import (
    accuracy_score,           # Overall accuracy
    confusion_matrix,         # True/False positives/negatives
    classification_report     # Precision, recall, F1-score
)

print("=" * 80)
print("STARTUP SURVIVAL PREDICTION MODEL")
print("=" * 80)
print()
print("Goal: Predict if a startup will succeed (1) or fail (0)")
print("Success = operating or acquired | Failure = closed")
print()

# STEP 2: LOAD FEATURE-ENGINEERED DATA
# -------------------------------------
# Load the data we created in feature_engineering.py
print("-" * 80)
print("LOADING DATA")
print("-" * 80)
print()

data_path = "features_data.csv"
df = pd.read_csv(data_path)

print(f"✓ Loaded {len(df):,} companies with {df.shape[1]} features")
print()

# STEP 3: PREPARE DATA FOR MODELING
# ----------------------------------
# We need to separate features (X) from target (y)
print("-" * 80)
print("PREPARING DATA FOR MODELING")
print("-" * 80)
print()

# Define which columns are features (predictors)
# We'll use all our engineered features
feature_columns = [
    # Funding features
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

# Keep only features that exist in our dataset
feature_columns = [col for col in feature_columns if col in df.columns]

# X = features (input data)
# What the model uses to make predictions
X = df[feature_columns].copy()

# y = target (what we want to predict)
# 1 = success (operating/acquired), 0 = failure (closed)
y = df['survival_target'].copy()

# Remove any rows where target is missing (shouldn't be any, but just in case)
valid_indices = y.notna()
X = X[valid_indices]
y = y[valid_indices]

print(f"Features (X): {X.shape[0]:,} companies × {X.shape[1]} features")
print(f"Target (y):   {len(y):,} labels")
print()

# Check for any missing values in features
missing_counts = X.isnull().sum().sum()
if missing_counts > 0:
    print(f"⚠ Warning: Found {missing_counts} missing values in features")
    print("  Filling with median values...")
    X = X.fillna(X.median())
    print("  ✓ Done")
    print()

# STEP 4: CHECK CLASS DISTRIBUTION
# ---------------------------------
# Important! If we have imbalanced classes (way more successes than failures),
# the model might just predict "success" for everything
print("-" * 80)
print("TARGET VARIABLE DISTRIBUTION")
print("-" * 80)
print()

# Count each class
class_counts = y.value_counts()
success_count = class_counts.get(1, 0)
failure_count = class_counts.get(0, 0)
total = success_count + failure_count

print(f"Success (1): {success_count:6,} companies ({success_count/total*100:5.2f}%)")
print(f"Failure (0): {failure_count:6,} companies ({failure_count/total*100:5.2f}%)")
print()

# Calculate imbalance ratio
imbalance_ratio = success_count / failure_count
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1 (success:failure)")

if imbalance_ratio > 2:
    print("⚠ Classes are imbalanced!")
    print("  Solution: We'll use class_weight='balanced' in our models")
    print("  This gives more weight to the minority class (failures)")
else:
    print("✓ Classes are reasonably balanced")

print()

# STEP 5: SPLIT DATA INTO TRAIN AND TEST SETS
# --------------------------------------------
# Why? We train the model on one set and test on another
# This prevents "overfitting" (memorizing instead of learning patterns)
print("-" * 80)
print("SPLITTING DATA: Train/Test")
print("-" * 80)
print()

# Split: 80% for training, 20% for testing
# random_state=42: Makes the split reproducible (same split every time)
# stratify=y: Keeps the same class ratio in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Keep same success/failure ratio
)

print(f"Training set:   {len(X_train):6,} companies ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set:       {len(X_test):6,} companies ({len(X_test)/len(X)*100:.1f}%)")
print()
print("Why this split?")
print("  - Train: Model learns patterns from this data")
print("  - Test: We evaluate how well the model generalizes to new data")
print()

# STEP 6: BUILD MODEL #1 - LOGISTIC REGRESSION
# ---------------------------------------------
# What is Logistic Regression?
# - A simple classification algorithm
# - Calculates probability that company will succeed (0 to 1)
# - If probability > 0.5, predicts success; otherwise failure
# - Shows which features are most important (via coefficients)
print("=" * 80)
print("MODEL #1: LOGISTIC REGRESSION")
print("=" * 80)
print()

print("What is Logistic Regression?")
print("  - Calculates probability of success (0 to 1)")
print("  - Simple, fast, interpretable")
print("  - Good baseline for classification problems")
print()

# Create the model
# class_weight='balanced': Handles class imbalance automatically
# max_iter=1000: Maximum training iterations (default might not be enough)
# random_state=42: For reproducibility
logistic_model = LogisticRegression(
    class_weight='balanced',  # Handle imbalanced classes
    max_iter=1000,            # Enough iterations to converge
    random_state=42           # Reproducibility
)

print("Training Logistic Regression model...")

# Train the model on training data
# This is where the "learning" happens!
logistic_model.fit(X_train, y_train)

print("✓ Training complete!")
print()

# STEP 7: EVALUATE LOGISTIC REGRESSION
# -------------------------------------
print("-" * 80)
print("EVALUATING: Logistic Regression")
print("-" * 80)
print()

# Make predictions on test set
# Remember: the model has never seen this data during training!
y_pred_logistic = logistic_model.predict(X_test)

# Also get probability predictions (useful to understand confidence)
y_pred_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]  # Probability of success

# METRIC 1: Accuracy
# ------------------
# What percentage of predictions were correct?
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Accuracy: {accuracy_logistic:.4f} ({accuracy_logistic*100:.2f}%)")
print(f"  → Out of 100 predictions, we got {accuracy_logistic*100:.1f} correct")
print()

# METRIC 2: Confusion Matrix
# ---------------------------
# Shows breakdown of predictions:
# - True Positives (TP): Correctly predicted success
# - True Negatives (TN): Correctly predicted failure
# - False Positives (FP): Predicted success, but actually failed
# - False Negatives (FN): Predicted failure, but actually succeeded
cm_logistic = confusion_matrix(y_test, y_pred_logistic)

print("Confusion Matrix:")
print()
print("                  Predicted")
print("                  Fail | Success")
print("         -----------------------")
print(f"Actual Fail    |  {cm_logistic[0][0]:4} | {cm_logistic[0][1]:4}  (TN | FP)")
print(f"       Success |  {cm_logistic[1][0]:4} | {cm_logistic[1][1]:4}  (FN | TP)")
print()

tn, fp, fn, tp = cm_logistic.ravel()
print("Interpretation:")
print(f"  True Negatives (TN):  {tn:4} - Correctly identified failures")
print(f"  False Positives (FP): {fp:4} - Wrongly predicted success (Type I error)")
print(f"  False Negatives (FN): {fn:4} - Wrongly predicted failure (Type II error)")
print(f"  True Positives (TP):  {tp:4} - Correctly identified successes")
print()

# METRIC 3: Classification Report
# --------------------------------
# Precision: Of all predicted successes, how many were actually successful?
# Recall: Of all actual successes, how many did we identify?
# F1-score: Harmonic mean of precision and recall (balanced metric)
print("Classification Report:")
print()
print(classification_report(y_test, y_pred_logistic, target_names=['Failure', 'Success']))

print("What these metrics mean:")
print("  Precision: If model says 'success', how often is it right?")
print("  Recall: Of all actual successes, how many did we catch?")
print("  F1-score: Balance between precision and recall")
print()

# STEP 8: FEATURE IMPORTANCE (LOGISTIC REGRESSION)
# -------------------------------------------------
# Which features matter most for predicting survival?
print("-" * 80)
print("FEATURE IMPORTANCE: Logistic Regression")
print("-" * 80)
print()

# Get coefficients (weights) for each feature
# Positive coefficient = feature increases success probability
# Negative coefficient = feature decreases success probability
coefficients = logistic_model.coef_[0]

# Create DataFrame for easy viewing
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

# Sort by absolute value (magnitude of impact)
feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)

print("Top 10 most important features:")
print()
print(f"{'Feature':<30} {'Coefficient':>12} {'Impact':>10}")
print("-" * 55)
for idx, row in feature_importance_df.head(10).iterrows():
    impact = "Positive" if row['Coefficient'] > 0 else "Negative"
    print(f"{row['Feature']:<30} {row['Coefficient']:>12.4f} {impact:>10}")

print()
print("How to read this:")
print("  Positive coefficient → Increases chance of success")
print("  Negative coefficient → Decreases chance of success")
print("  Larger magnitude → Stronger effect")
print()

# STEP 9: BUILD MODEL #2 - DECISION TREE
# ---------------------------------------
# What is a Decision Tree?
# - Makes decisions using a series of yes/no questions
# - Easy to visualize and understand
# - Shows exactly which rules lead to success/failure
print("=" * 80)
print("MODEL #2: DECISION TREE")
print("=" * 80)
print()

print("What is a Decision Tree?")
print("  - Makes decisions using if/else rules")
print("  - Very intuitive and easy to explain")
print("  - Can capture non-linear patterns")
print()

# Create the model
# max_depth=5: Limit tree depth to prevent overfitting
# class_weight='balanced': Handle class imbalance
# random_state=42: Reproducibility
tree_model = DecisionTreeClassifier(
    max_depth=5,              # Shallow tree to prevent overfitting
    class_weight='balanced',  # Handle imbalanced classes
    random_state=42           # Reproducibility
)

print("Training Decision Tree model...")

# Train the model
tree_model.fit(X_train, y_train)

print("✓ Training complete!")
print()

# STEP 10: EVALUATE DECISION TREE
# --------------------------------
print("-" * 80)
print("EVALUATING: Decision Tree")
print("-" * 80)
print()

# Make predictions
y_pred_tree = tree_model.predict(X_test)

# METRIC 1: Accuracy
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Accuracy: {accuracy_tree:.4f} ({accuracy_tree*100:.2f}%)")
print(f"  → Out of 100 predictions, we got {accuracy_tree*100:.1f} correct")
print()

# METRIC 2: Confusion Matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)

print("Confusion Matrix:")
print()
print("                  Predicted")
print("                  Fail | Success")
print("         -----------------------")
print(f"Actual Fail    |  {cm_tree[0][0]:4} | {cm_tree[0][1]:4}  (TN | FP)")
print(f"       Success |  {cm_tree[1][0]:4} | {cm_tree[1][1]:4}  (FN | TP)")
print()

# METRIC 3: Classification Report
print("Classification Report:")
print()
print(classification_report(y_test, y_pred_tree, target_names=['Failure', 'Success']))

# STEP 11: FEATURE IMPORTANCE (DECISION TREE)
# --------------------------------------------
print("-" * 80)
print("FEATURE IMPORTANCE: Decision Tree")
print("-" * 80)
print()

# Decision trees calculate importance differently than logistic regression
# Based on how much each feature reduces impurity (disorder) in the data
importances = tree_model.feature_importances_

# Create DataFrame
feature_importance_tree = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importances
})

# Sort by importance
feature_importance_tree = feature_importance_tree.sort_values('Importance', ascending=False)

print("Top 10 most important features:")
print()
print(f"{'Feature':<30} {'Importance':>12} {'%':>8}")
print("-" * 52)
for idx, row in feature_importance_tree.head(10).iterrows():
    pct = row['Importance'] * 100
    print(f"{row['Feature']:<30} {row['Importance']:>12.4f} {pct:>7.2f}%")

print()
print("How to read this:")
print("  Higher importance → Feature is used more in tree decisions")
print("  All importances sum to 1.0 (100%)")
print()

# STEP 12: MODEL COMPARISON
# --------------------------
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print()

print(f"{'Model':<25} {'Accuracy':>10} {'Best For':<30}")
print("-" * 70)
print(f"{'Logistic Regression':<25} {accuracy_logistic:>10.4f} {'Linear relationships, interpretable coefficients':<30}")
print(f"{'Decision Tree':<25} {accuracy_tree:>10.4f} {'Non-linear patterns, visual rules':<30}")
print()

if accuracy_logistic > accuracy_tree:
    print(f"✓ Logistic Regression performs better ({accuracy_logistic:.4f} vs {accuracy_tree:.4f})")
elif accuracy_tree > accuracy_logistic:
    print(f"✓ Decision Tree performs better ({accuracy_tree:.4f} vs {accuracy_logistic:.4f})")
else:
    print("✓ Both models perform equally well")

print()

# STEP 13: BUSINESS INSIGHTS
# ---------------------------
print("=" * 80)
print("BUSINESS INSIGHTS FOR PE/VC")
print("=" * 80)
print()

print("What makes a startup more likely to succeed?")
print()

# Get top positive features from logistic regression
top_positive = feature_importance_df[feature_importance_df['Coefficient'] > 0].head(5)
print("Success factors (increase survival probability):")
for idx, row in top_positive.iterrows():
    print(f"  ✓ {row['Feature']}")
print()

# Get top negative features
top_negative = feature_importance_df[feature_importance_df['Coefficient'] < 0].head(5)
print("Risk factors (decrease survival probability):")
for idx, row in top_negative.iterrows():
    print(f"  ⚠ {row['Feature']}")
print()

print("Model Performance Summary:")
print(f"  - We can predict startup survival with ~{max(accuracy_logistic, accuracy_tree)*100:.1f}% accuracy")
print(f"  - This is {max(accuracy_logistic, accuracy_tree)/0.5:.1f}x better than random guessing")
print(f"  - Model is ready for due diligence and portfolio risk assessment")
print()

print("=" * 80)
print("SURVIVAL PREDICTION MODEL COMPLETE!")
print("=" * 80)
print()
print("Next script to run: model_series_a.py")
print()
