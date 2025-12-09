# Startup Financial Health Predictor

A complete machine learning project for predicting startup success and Series A funding using real Crunchbase data. Perfect for PE/VC portfolio analysis!

## 📋 Project Overview

This project builds two prediction models:
1. **Startup Survival Predictor** - Predicts if a startup will succeed (operating/acquired) or fail (closed)
2. **Series A Predictor** - Predicts if a startup will raise Series A funding

**Built with:** Simple, interpretable models (Logistic Regression & Decision Trees)
**Dataset:** 54,295 startup companies from Crunchbase
**Educational focus:** Every line of code is heavily commented to help you learn!

## 🎯 Why This Project?

- ✅ Real-world startup data (Crunchbase)
- ✅ Business-focused predictions (PE/VC relevant)
- ✅ Complete end-to-end ML pipeline
- ✅ Simple, explainable models
- ✅ Perfect for portfolios and interviews

## 📁 Project Structure

```
data science project/
├── ds.py                      # Downloads the dataset
├── investments_VC.csv         # Raw dataset (auto-downloaded)
│
├── data_exploration.py        # Step 1: Explore the data
├── data_cleaning.py           # Step 2: Clean messy data
├── feature_engineering.py     # Step 3: Create predictive features
├── model_survival.py          # Step 4: Predict startup success
├── model_series_a.py          # Step 5: Predict Series A funding
│
├── cleaned_data.csv           # Output from step 2
├── features_data.csv          # Output from step 3
└── README.md                  # This file
```

## 🚀 How to Run

### Step 0: Install Required Libraries

```bash
pip install pandas numpy scikit-learn kagglehub
```

All these should already be in Anaconda!

### Step 1: Download the Dataset

```bash
python ds.py
```

This downloads the Crunchbase startup investment dataset (~54K companies).

### Step 2: Explore the Data

```bash
python data_exploration.py
```

**What you'll learn:**
- Dataset structure and size
- Missing values analysis
- Target variable distributions
- Funding statistics
- Geographic and industry insights

**Time:** ~30 seconds
**Output:** Detailed statistics printed to console

### Step 3: Clean the Data

```bash
python data_cleaning.py
```

**What happens:**
- Removes commas from funding amounts ("1,750,000" → 1750000)
- Converts dates to proper format
- Creates target variables (survival_target, series_a_target)
- Handles missing values
- Saves to `cleaned_data.csv`

**Time:** ~1-2 minutes
**Output:** `cleaned_data.csv` with ~48K companies

### Step 4: Engineer Features

```bash
python feature_engineering.py
```

**What happens:**
- Creates 18 new predictive features
- Temporal features (company age, funding timing)
- Funding features (has_seed, avg_funding_per_round)
- Geographic features (is_usa, is_tech_hub)
- Industry features (is_software, is_biotech)
- Saves to `features_data.csv`

**Time:** ~1-2 minutes
**Output:** `features_data.csv` with all features

### Step 5: Build Survival Model

```bash
python model_survival.py
```

**What happens:**
- Builds Logistic Regression model
- Builds Decision Tree model
- Evaluates both models (accuracy, precision, recall)
- Shows feature importance
- Provides business insights

**Time:** ~30 seconds
**Expected Accuracy:** 75-80%

### Step 6: Build Series A Model

```bash
python model_series_a.py
```

**What happens:**
- Filters to funded startups only
- Builds Logistic Regression model
- Builds Decision Tree model
- Compares performance
- Provides VC-specific insights

**Time:** ~30 seconds
**Expected Accuracy:** 70-75%

## 📊 What You'll Learn

### Data Science Skills
- ✅ Data exploration and understanding
- ✅ Data cleaning (70% of the work!)
- ✅ Feature engineering from business metrics
- ✅ Binary classification modeling
- ✅ Model evaluation (accuracy, precision, recall, F1)
- ✅ Handling class imbalance
- ✅ Feature importance analysis

### Business Understanding
- ✅ What factors predict startup success?
- ✅ How to identify Series A candidates
- ✅ Red flags and success signals
- ✅ Geographic and industry patterns
- ✅ Funding stage progression

### Technical Skills
- ✅ Pandas for data manipulation
- ✅ NumPy for numerical operations
- ✅ Scikit-learn for machine learning
- ✅ Model interpretation and explanation

## 🎓 Educational Features

Every script includes:
- **Extensive comments** explaining WHY, not just WHAT
- **Step-by-step breakdown** of each operation
- **Business context** for PE/VC relevance
- **Technical term definitions** in plain English
- **Expected outputs** at each stage

## 📈 Expected Results

### Survival Model
- **Accuracy:** ~75-80%
- **Dataset:** 48K companies
- **Target:** 83% success, 17% failure (imbalanced)
- **Best for:** Risk assessment, portfolio monitoring

### Series A Model
- **Accuracy:** ~70-75%
- **Dataset:** ~30K funded companies
- **Target:** ~10% raise Series A (highly imbalanced)
- **Best for:** Deal flow prioritization, investment decisions

## 🔍 Key Insights You'll Discover

The models will reveal:
- Which funding patterns predict success
- Importance of geography (tech hubs vs. others)
- How company age affects outcomes
- Industry-specific success rates
- Timing factors in fundraising

## 💼 Portfolio Applications

### For VCs/PEs:
1. **Pipeline Scoring** - Rank incoming deals
2. **Due Diligence** - Identify risk factors
3. **Portfolio Monitoring** - Track company health
4. **Market Insights** - Understand success patterns

## 🐛 Troubleshooting

### Issue: "No module named 'kagglehub'"
**Solution:** Run `pip install kagglehub`

### Issue: "File not found: investments_VC.csv"
**Solution:** Run `python ds.py` first to download the dataset

### Issue: "SettingWithCopyWarning"
**Solution:** These are warnings, not errors. The code will still work!

### Issue: Low model accuracy
**Expected:** 70-80% is GOOD for this problem! Predicting startup success is inherently difficult.

## 📝 Next Steps (Optional)

Want to go further? Try:

1. **Add visualizations**
   - Feature importance plots
   - ROC curves
   - Confusion matrix heatmaps

2. **Try advanced models**
   - Random Forest
   - XGBoost
   - Neural Networks

3. **Hyperparameter tuning**
   - GridSearchCV
   - RandomizedSearchCV

4. **Build a dashboard**
   - Streamlit web app
   - Flask API
   - Interactive predictions

5. **Time-based prediction**
   - Predict failure within 2 years
   - Time to Series A

## 📚 Learning Resources

- **Logistic Regression:** [Scikit-learn Docs](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- **Decision Trees:** [Scikit-learn Docs](https://scikit-learn.org/stable/modules/tree.html)
- **Model Evaluation:** [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Class Imbalance:** [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)

## 🤝 Questions?

If you don't understand something in the code:
1. Read the comments carefully - they explain every step!
2. Run the scripts step by step
3. Print intermediate results to see what's happening
4. Google unfamiliar terms - learning is part of the process!

## 📄 License

This is an educational project using publicly available Crunchbase data.

---

**Built with ❤️ for learning data science and understanding startup dynamics!**

Good luck with your PE/VC career! 🚀
