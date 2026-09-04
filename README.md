# Ola Driver Churn Prediction

This project builds a predictive model to determine driver churn risk, drawing
on demographic details (city, age, gender), tenure information (joining date,
last working date), and historical performance data (quarterly ratings,
monthly business value, grade, and income).

The dataset is a monthly panel of Ola driver records for 2019–2020. The
notebook aggregates this into a driver-level dataset, engineers churn-relevant
features, handles missing values via KNN imputation, and addresses class
imbalance before training two ensemble models — a Bagging approach
(Random Forest) and a Boosting approach (XGBoost) — to predict attrition and
surface actionable retention recommendations.

## Dataset Overview

- **19,104 rows** (monthly panel) → aggregated to **2,381 unique drivers**
- **13 original columns**: demographics, tenure, and performance attributes
- **Overall churn rate: ~68%** (1,616 churned drivers vs. 765 still active)
- **Source**: Ola driver dataset, 2019–2020 ([`ola_driver.csv`](https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/002/492/original/ola_driver_scaler.csv))

| Column | Description |
|---|---|
| `MMM-YY` | Reporting date (monthly) |
| `Driver_ID` | Unique driver ID |
| `Age` | Driver's age |
| `Gender` | Male = 0, Female = 1 |
| `City` | City code |
| `Education_Level` | 0 = 10+, 1 = 12+, 2 = Graduate |
| `Income` | Monthly average income |
| `Dateofjoining` | Joining date |
| `LastWorkingDate` | Last working date (if churned) |
| `Joining Designation` | Designation at time of joining |
| `Grade` | Grade at time of reporting |
| `Total Business Value` | Business value acquired that month (negative = cancellations/refunds/EMI adjustments) |
| `Quarterly Rating` | Rating from 1 (lowest) to 5 (highest) |

## Methodology / Approach

1. **Exploratory Data Analysis** — structure, data types, missing values, univariate and bivariate distributions.
2. **Driver-level aggregation** — collapsed the monthly panel into one row per driver via `groupby`.
3. **Feature engineering**:
   - `target` — 1 if the driver has a `LastWorkingDate` (i.e. churned), else 0
   - `quarterly_rating_increased` — 1 if a driver's rating rose over time
   - `income_increased` — 1 if a driver's income rose over time
   - `grade_increased` — 1 if a driver was promoted a grade
   - `tenure_months` — months from joining to last activity
4. **KNN Imputation** — filled missing `Age` / `Gender` values using nearest-neighbor similarity on numeric features.
5. **One-hot encoding** — applied to `City` and `Education_Level`.
6. **Class imbalance treatment** — SMOTE oversampling on the training set only.
7. **Standardization** — `StandardScaler` applied to numeric features.
8. **Modeling** — Random Forest (Bagging) and XGBoost (Boosting), both tuned via `GridSearchCV`.

## Key EDA Insights

- `Total Business Value` is heavily right-skewed with extreme outliers.
- Churned drivers skew younger and have noticeably shorter tenure.
- A `Quarterly Rating` of 1 correlates strongly with a much higher churn rate.
- Income growth is one of the strongest protective factors against churn.
- Churn rate varies substantially by city, pointing to local market effects.

## Model Results

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy |
|---|---|---|---|
| Random Forest (Bagging) | 0.9414 | ~0.90 | ~85% |
| XGBoost (Boosting) | 0.9424 | ~0.90+ | ~85% |

- Both models show strong precision/recall on the "Churned" class after SMOTE correction.
- `Quarterly Rating` is by far the most important feature, followed by tenure and income-related signals.

## Actionable Recommendations

- **Deploy a real-time churn risk score** to flag high-risk drivers before they leave.
- **Front-load retention support** in the first few months — tenure is the highest-risk period.
- **Review compensation proactively** — income growth is strongly protective, but few drivers currently experience it.
- **Treat a rating drop to 1 as an early-warning trigger** for coaching or incentives.
- **Design city-specific interventions**, since churn drivers differ significantly by local market.

## Repo Contents

- `Ola_Driver_Churn_Prediction.ipynb` — full, executed analysis notebook (EDA → feature engineering → modeling → evaluation → insights)

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
jupyter notebook Ola_Driver_Churn_Prediction.ipynb
```

## Notes

- The dataset used here was sourced from a public GitHub mirror of the original
  Ola driver dataset, since the original CloudFront-hosted CSV was not directly
  reachable in the analysis environment. Row counts, unique driver counts, and
  churn totals were verified against the original dataset description.
