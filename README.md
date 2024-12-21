![banner](https://github.com/PJURC-data-science/retail-sales-forecasting/blob/main/media/banner.png)

# Retail Sales Forecasting: Annual Revenue Prediction
[View Dashboard](https://public.tableau.com/app/profile/pijus.jur.iukonis/viz/stores_sales/Dashboard1)
[View Notebook](https://github.com/PJURC-data-science/retail-sales-forecasting/blob/main/Retail%20Sales%20Forecasting.ipynb)

A comprehensive analysis to forecast retail chain sales for 2013 using historical data and multiple modeling approaches. This study provides sales predictions and performance metrics to support CFO's financial forecasting requirements.

## Overview

### Business Question 
Can we accurately forecast weekly sales for each store from 2012-12-10 to 2013-12-10 to support annual financial planning?

### Key Findings
- Total forecast: $2,635.28M
- Average store growth: ~7%
- Model accuracy: 1.47% MAPE
- RMSE: $878,438
- Department data crucial for accuracy

### Impact/Results
- Delivered annual forecast
- Identified growth patterns
- Quantified forecast reliability
- Created visualization dashboard
- Established store performance metrics

## Data

### Source Information
- Dataset: US Retail Chain Sales
- Period: 2010-2012 historical
- Scope: Store-level weekly sales
- Coverage: Multiple stores/departments
- Additional: Economic indicators

### Variables Analyzed
- Weekly sales
- Store characteristics
- Department data
- Economic indicators:
  - Temperature
  - Fuel prices
  - CPI
  - Unemployment
- Promotional data
- Holiday flags

## Methods

### Analysis Approach
1. Data Engineering
   - Time-based features
   - Seasonal multipliers
   - Holiday patterns
   - Feature importance analysis
2. Model Development
   - Multiple iterations
   - Model comparison
   - Ensemble testing
3. Performance Analysis
   - Store-level metrics
   - Chain-wide predictions
   - Visualization creation

### Tools Used
- Python (Data Science)
  - Primary Models:
    - GradientBoosting (best performer)
    - XGBoost
    - LightGBM
    - CatBoost
    - ElasticNet
  - Ensemble Methods:
    - Weighted Prediction Blending
    - Stacking Regressor
    - Multi-Level Stacking
  - Feature Engineering:
    - Mutual Information
    - PhiK correlation
    - Time-based features
  - Performance Metrics:
    - MAPE: 1.47%
    - RMSE: $878,438
  - Visualization:
    - Tableau Public
    - Matplotlib/Seaborn

## Getting Started

### Prerequisites
```python
joblib==1.4.2
lightgbm==4.5.0
matplotlib==3.8.4
numpy==2.2.0
pandas==2.2.3
phik==0.12.4
scikit_learn==1.6.0
scipy==1.14.1
seaborn==0.13.2
statsmodels==0.14.4
xgboost==2.1.3
ipython==8.12.3
```

### Installation & Usage
```bash
git clone git@github.com:TuringCollegeSubmissions/pjurci-STCI.2.1.git
cd git@github.com:PJURC-data-science/retail-sales-forecasting.git
pip install -r requirements.txt
jupyter notebook "Retail Sales Forecasting.ipynb"
```

## Project Structure
```
credit-risk-prediction/
│   README.md
│   requirements.txt
│   Retail Sales Forecasting.ipynb
|   utils.py
|   styles.css
|   stores_sales.twb
└── data/
    └── features.csv
    └── neg_markdown_date.csv
    └── sales.csv
    └── stores.csv
└── exports/
    └── df_raw.csv
    └── stores_metrics.csv
    └── stores_sales.csv
└── models/
    └── catboost_tuned.joblib
    └── elasticnet_tuned.joblib
    └── gradient_boosting_tuned.joblib
    └── lightgbm_tuned_tuned.joblib
    └── xgboost_tuned.joblib
```

## Strategic Recommendations
1. **Store Management**
   - Monitor outlier stores
   - Address declining locations
   - Support growth patterns
   - Track department performance

2. **Forecast Application**
   - Update quarterly
   - Use moving windows
   - Segment store categories
   - Monitor accuracy

3. **Data Strategy**
   - Add location data
   - Expand feature set
   - Track department metrics
   - Monitor external factors

## Future Improvements
- Implement specialized models (Prophet, Chronos T5)
- Add location-based features
- Expand department analysis
- Create store categories
- Update more frequently
- Add external forecasts
- Enhance feature engineering
- Test SARIMAX implementation
- Expand hyperparameter tuning