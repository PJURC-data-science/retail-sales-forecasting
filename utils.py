import re
import tkinter as tk
from typing import Dict, List
import warnings
import lightgbm
import matplotlib.gridspec as gridspec
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import time
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import phik


import numpy as np
import pandas as pd
from xgboost import XGBRegressor

RANDOM_STATE = 98
DECAY_RATE = 0.3
COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
EXPORT_FOLDER = "exports"
DATA_FOLDER = "data"
DATATYPE_COLOR_MAP = {
    "numeric": "#1f77b4",
    "categorical": "#ff7f0e",
}
TRAIN_START = '2010-02-05'
TRAIN_END = '2012-06-10'
VALIDATION_START = '2012-06-10'
VALIDATION_END = '2012-12-10'
FORECAST_START = '2012-12-10'
FORECAST_END = '2013-12-10'


class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        """
        Initialize the MeanEncoder with columns to encode.

        Parameters
        ----------
        columns : list of str, optional
            The columns to encode. If None, all categorical columns
            are used. Defaults to None.
        """

        self.columns = columns
        self.mean_encodings = {}

    def fit(self, X, y):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object", "category"]).columns

        for col in self.columns:
            mean_encoding = y.groupby(X[col]).mean()
            self.mean_encodings[col] = mean_encoding
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col, encoding in self.mean_encodings.items():
            X_encoded[col] = X_encoded[col].map(encoding)
            X_encoded[col] = X_encoded[col].fillna(encoding.mean())
        return X_encoded
    
    
def get_screen_width() -> int:
    """Retrieves the screen width using a tkinter root window and returns the screen width value."""
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    root.destroy()

    return screen_width


def set_font_size() -> dict:
    """Sets the font sizes for visualization elements based on the screen width."""
    base_font_size = round(get_screen_width() / 100, 0)
    font_sizes = {
        "font.size": base_font_size * 0.6,
        "axes.titlesize": base_font_size * 0.4,
        "axes.labelsize": base_font_size * 0.6,
        "xtick.labelsize": base_font_size * 0.4,
        "ytick.labelsize": base_font_size * 0.4,
        "legend.fontsize": base_font_size * 0.6,
        "figure.titlesize": base_font_size * 0.6,
    }

    return font_sizes


def custom_format(x: float) -> str:
    """
    Formats a given number to a string with a specific decimal precision.

    Args:
        x (float): The number to be formatted.

    Returns:
        str: The formatted number as a string. If the number is an integer, it is formatted as an integer with no decimal places.
        Otherwise, it is formatted with two decimal places.
    """
    if x == int(x):
        return "{:.0f}".format(x)
    else:
        return "{:.2f}".format(x)
    

def check_duplicates(df: pd.DataFrame, df_name: str) -> None:
    """
    Check for duplicate rows in a pandas DataFrame and print the results.

    Args:
        df (pandas.DataFrame): The DataFrame to check for duplicates.
        df_name (str): The name of the DataFrame for printing purposes.

    Returns:
        None
    """
    duplicate_count = df.duplicated().sum()
    print(f"DataFrame: {df_name}")
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {duplicate_count}\n")
    duplicates = df[df.duplicated(keep=False)]
    sorted_duplicates = duplicates.sort_values(by=list(df.columns))
    sorted_duplicates[:10] if len(duplicates) > 0 else None


def boolean_analysis(df: pd.DataFrame, boolean_columns: List[str]) -> pd.DataFrame:
    """
    Analyze a boolean column in a DataFrame and return a DataFrame with the count, null count, true count, false count, true percentage, and false percentage.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the column to be analyzed.
    boolean_column : str
        The name of the boolean column to be analyzed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the analysis of the boolean column.
    """
    if isinstance(boolean_columns, str):
        boolean_columns = [boolean_columns]

    results = []
    for column in boolean_columns:
        analysis = {
            "column": column,
            "count": df[column].count(),
            "null_count": df[column].isnull().sum(),
            "true_count": df[column].sum(),
            "false_count": (~df[column]).sum(),
            "true_percentage": df[column].mean() * 100,
            "false_percentage": (1 - df[column].mean()) * 100,
        }
        results.append(analysis)

    return pd.DataFrame(results).set_index("column")


def datetime_analysis(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    """
    Analyze datetime columns in a DataFrame and return analysis including transaction counts per month.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to be analyzed.
    datetime_columns : List[str] or str
        The name(s) of the datetime column(s) to be analyzed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the analysis of the datetime columns.
    """
    if isinstance(datetime_columns, str):
        datetime_columns = [datetime_columns]

    results = []
    for column in datetime_columns:
        # Basic datetime statistics
        basic_analysis = {
            "column": column,
            "count": df[column].count(),
            "min": df[column].min(),
            "max": df[column].max(),
            "range": df[column].max() - df[column].min(),
            "mode": df[column].mode().iloc[0] if not df[column].mode().empty else None,
            "null_count": df[column].isnull().sum(),
            "unique_count": df[column].nunique(),
        }
        results.append(basic_analysis)

    # Create DataFrame and format dates
    result_df = pd.DataFrame(results).set_index("column")

    # Format datetime columns
    for col in ["min", "max"]:
        if col in result_df.columns:
            result_df[col] = result_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    return result_df


def missing_values(df: pd.DataFrame, missing_only=False) -> pd.DataFrame:
    """Returns a DataFrame that summarizes missing values in the train and test datasets, including column data types, sorted by column data type."""
    missing_values = round(df.isnull().sum(), 0)
    missing_values_perc = round((missing_values / len(df)) * 100, 1)
    column_data_types = df.dtypes

    missing_values = pd.DataFrame(
        {
            "Data Type": column_data_types,
            "Count #": missing_values,
            "Perc %": missing_values_perc,
        }
    )

    missing_values = missing_values.sort_values(by="Perc %", ascending=False)

    # Filter features with missing values
    if missing_only:
        missing_values = missing_values[(missing_values["Count #"] > 0)]

    return missing_values


def find_and_analyze_infinite_values(df: pd.DataFrame) -> None:
    """
    Check if a pandas Series contains infinite values.

    Args:
        x (pd.Series): The input Series

    Returns:
        pd.Series: A Series of the same shape as the input, with True values indicating infinite values and False otherwise
    """

    def _is_infinite(x):
        if pd.api.types.is_numeric_dtype(x):
            return np.isinf(x)
        else:
            return pd.Series(False, index=x.index)

    # Find rows with infinite values
    infinite_mask = df.apply(_is_infinite)
    infinite_rows = df[infinite_mask.any(axis=1)]

    # Count infinite values per feature
    infinite_counts = infinite_mask.sum()
    features_with_infinites = infinite_counts[infinite_counts > 0]

    # Collect information about infinite values
    infinite_info = {
        "rows": infinite_rows,
        "features": features_with_infinites.to_dict(),
        "total_infinites": infinite_mask.sum().sum(),
    }

    # Print results
    if infinite_rows.empty:
        print("No rows with infinite values found.")
    else:
        print(f"Found {len(infinite_rows)} row(s) with infinite values.")
        print("\nFeatures with infinite values:")
        for feature, count in features_with_infinites.items():
            print(f"  {feature}: {count} infinite value(s)")
        print(f"\nTotal number of infinite values: {infinite_info['total_infinites']}")


def get_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing the data types of each column in the input DataFrame, sorted by data type.

    Args:
        df (pandas.DataFrame): The input DataFrame for which to determine column data types.

    Returns:
        pandas.DataFrame: A DataFrame with column names as the index and a single column 'dtype' indicating
                        the data type of each column, sorted by data type.
    """
    col_dtypes = {}
    for col in df.columns:
        col_dtypes[col] = df[col].dtype
    col_dtypes = {
        k: v for k, v in sorted(col_dtypes.items(), key=lambda item: str(item[1]))
    }
    df_dtypes = pd.DataFrame(col_dtypes, index=["dtype"]).T

    return df_dtypes


def plot_seasonal_sales(df):
    """
    Create an enhanced sales plot with seasonal colors properly filled below the line.
    
    Parameters:
    df: DataFrame with 'Date' and 'Weekly_Sales' columns
    """
    # Create continuous date range to ensure no gaps
    monthly_sales = df.groupby(['Date'])['Weekly_Sales'].sum()
    date_range = pd.date_range(start=monthly_sales.index.min(),
                             end=monthly_sales.index.max(),
                             freq='D')
    monthly_sales = monthly_sales.reindex(date_range).fillna(method='ffill')
    monthly_sales = monthly_sales.reset_index()
    monthly_sales.columns = ['Date', 'Weekly_Sales']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot the line first
    ax.plot(monthly_sales['Date'], monthly_sales['Weekly_Sales'], 
           color=COLOR_PALETTE[0], linewidth=2, zorder=2)
    
    # Plot each season separately with exact date ranges
    for year in monthly_sales['Date'].dt.year.unique():
        # Winter (Dec previous year - Feb)
        winter_start = pd.Timestamp(f'{year-1}-12-01')
        winter_end = pd.Timestamp(f'{year}-02-28')
        mask = (monthly_sales['Date'] >= winter_start) & (monthly_sales['Date'] <= winter_end)
        if any(mask):
            ax.fill_between(monthly_sales['Date'][mask], 
                          0, monthly_sales['Weekly_Sales'][mask],
                          color='#A5F2F3', alpha=0.6, label='Winter' if year == monthly_sales['Date'].dt.year.min() else "")
        
        # Spring (Mar - May)
        spring_start = pd.Timestamp(f'{year}-03-01')
        spring_end = pd.Timestamp(f'{year}-05-31')
        mask = (monthly_sales['Date'] >= spring_start) & (monthly_sales['Date'] <= spring_end)
        if any(mask):
            ax.fill_between(monthly_sales['Date'][mask], 
                          0, monthly_sales['Weekly_Sales'][mask],
                          color='#D4F2CE', alpha=0.6, label='Spring' if year == monthly_sales['Date'].dt.year.min() else "")
        
        # Summer (Jun - Aug)
        summer_start = pd.Timestamp(f'{year}-06-01')
        summer_end = pd.Timestamp(f'{year}-08-31')
        mask = (monthly_sales['Date'] >= summer_start) & (monthly_sales['Date'] <= summer_end)
        if any(mask):
            ax.fill_between(monthly_sales['Date'][mask], 
                          0, monthly_sales['Weekly_Sales'][mask],
                          color='#FFE5B4', alpha=0.6, label='Summer' if year == monthly_sales['Date'].dt.year.min() else "")
        
        # Autumn (Sep - Nov)
        autumn_start = pd.Timestamp(f'{year}-09-01')
        autumn_end = pd.Timestamp(f'{year}-11-30')
        mask = (monthly_sales['Date'] >= autumn_start) & (monthly_sales['Date'] <= autumn_end)
        if any(mask):
            ax.fill_between(monthly_sales['Date'][mask], 
                          0, monthly_sales['Weekly_Sales'][mask],
                          color='#FFB6A3', alpha=0.6, label='Autumn' if year == monthly_sales['Date'].dt.year.min() else "")
        
        # Winter (Dec current year)
        winter_start = pd.Timestamp(f'{year}-12-01')
        winter_end = pd.Timestamp(f'{year}-12-31')
        mask = (monthly_sales['Date'] >= winter_start) & (monthly_sales['Date'] <= winter_end)
        if any(mask):
            ax.fill_between(monthly_sales['Date'][mask], 
                          0, monthly_sales['Weekly_Sales'][mask],
                          color='#A5F2F3', alpha=0.6, label='Winter' if year == monthly_sales['Date'].dt.year.min() else "")
    
    # Customize x-axis
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    plt.xticks(rotation=0)
    
    # Add title and labels
    plt.title('Weekly Sales by Month', pad=20, size=14, weight='bold')
    plt.xlabel('Month')
    plt.ylabel('Weekly Sales ($)')
    
    # Add legend
    plt.legend(title='Seasons')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def plot_holiday_sales(df):
    """
    Create a sales plot with holiday weeks highlighted and step-style line.
    
    Parameters:
    df: DataFrame with Date, Weekly_Sales and IsHoliday columns
    """
    # Get unique dates and their total sales
    unique_dates = df.groupby('Date', as_index=False).agg({
        'Weekly_Sales': 'sum',
        'IsHoliday': 'first'
    })
    
    # Sort by date
    unique_dates = unique_dates.sort_values('Date')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Plot the sales line as steps
    ax.step(unique_dates['Date'], unique_dates['Weekly_Sales'], 
            where='post',  # 'post' makes the steps extend to the right
            color='#8884d8', linewidth=2, zorder=3, label='Weekly Sales')
    
    # Highlight holiday weeks one at a time
    holiday_dates = unique_dates[unique_dates['IsHoliday'] == True]
    
    for idx, row in holiday_dates.iterrows():
        ax.fill_between([row['Date'], row['Date'] + pd.Timedelta(days=6)],  # one week width
                       0,
                       [row['Weekly_Sales'], row['Weekly_Sales']],  # constant sales value
                       color='red', alpha=0.3)
    
    # Add holiday week label to legend
    ax.fill_between([], [], [], color='red', alpha=0.3, label='Holiday Week')
    
    # Customize x-axis
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    plt.xticks(rotation=0)
    
    # Add title and labels
    plt.title('Weekly Sales with Holiday Weeks Highlighted', pad=20, size=14, weight='bold')
    plt.xlabel('Month')
    plt.ylabel('Weekly Sales ($)')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    plt.grid(True, alpha=0.3, zorder=1)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()


def create_retail_features(df):
    """
    Create features for retail sales prediction with optimized performance
    for large datasets
    """
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    sales_array = data['Weekly_Sales'].values
    
    # Basic Date Features
    data['Year'] = data['Date'].dt.year.astype(int)
    data['Month'] = data['Date'].dt.month.astype(int)
    data['Week'] = data['Date'].dt.isocalendar().week.astype(int)
    data['Quarter'] = data['Date'].dt.quarter.astype(int)

    # Cyclical Date Features
    data['Month_sin'] = np.sin(2 * np.pi * data['Month']/12)
    data['Week_sin'] = np.sin(2 * np.pi * data['Week']/52)
    data['Quarter_sin'] = np.sin(2 * np.pi * data['Quarter']/4)
    
    # Historical Metrics & moving averages
    data['running_count'] = data.groupby(['Store', 'Dept']).cumcount() + 1
    data['hist_cum_sales'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].cumsum()
    data['hist_avg_sales'] = (data['hist_cum_sales'] - data['Weekly_Sales']) / \
                            (data['running_count'] - 1)
    for window in [4, 8, 12]:
        data[f'rolling_mean_{window}w'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
    data['ewm_sales_4w'] = pd.Series(sales_array).ewm(span=4, adjust=False).mean().shift(1)
    data['ewm_sales_12w'] = pd.Series(sales_array).ewm(span=12, adjust=False).mean().shift(1)

    # Lag features
    # 1 week, 4 weeks, 52 weeks
    lags = [1, 4, 52, 104]
    for lag in lags:
        data[f'sales_lag_{lag}'] = np.concatenate((
            np.full(lag, np.nan), sales_array[:-lag]
        ))

    # YTD calculations
    def _ytd_mean(x):
        sort_idx = x.index[np.argsort(data.loc[x.index, 'Date'])]
        x = x[sort_idx]
        return x.expanding().mean().shift(1)
    data['ytd_sales'] = data.groupby(['Store', 'Dept', 'Year'])['Weekly_Sales'].transform(_ytd_mean)
    
    # Markdown features
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    data['total_markdown'] = data[markdown_cols].sum(axis=1)
    data['hist_avg_markdown'] = data.groupby(['Store', 'Dept'])['total_markdown'].transform(
        lambda x: x.expanding().mean().shift(1))
    data['promo_intensity'] = data[markdown_cols].sum(axis=1)
    
    # Growth rates
    for lag_col in ['sales_lag_1', 'sales_lag_4', 'sales_lag_52']:
        growth_col = f'growth_{lag_col}'
        denominator = data[lag_col].replace(0, 1e-10)  
        data[growth_col] = (data['Weekly_Sales'] - data[lag_col]) / denominator
        data[growth_col] = data[growth_col].clip(-10, 10)
    
    # External factors
    data['temp_mean'] = data.groupby('Store')['Temperature'].transform(
        lambda x: x.expanding().mean().shift(1))
    data['temp_std'] = data.groupby('Store')['Temperature'].transform(
        lambda x: x.expanding().std().shift(1))
    data['temp_zscore'] = (data['Temperature'] - data['temp_mean']) / data['temp_std']
    data['fuel_price_ma4'] = data.groupby('Store')['Fuel_Price'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
    data['cpi_change'] = data.groupby('Store')['CPI'].transform('pct_change')
    data['unemployment_change'] = data.groupby('Store')['Unemployment'].transform('pct_change')
    
    # Holiday features
    def _holiday_sales_mean(x):
        mask = data.loc[x.index, 'IsHoliday']
        result = x.copy()
        result[~mask] = np.nan
        return result.expanding().mean().shift(1)
    data['hist_holiday_sales'] = data.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(_holiday_sales_mean)
    
    # Store metrics
    data['sales_per_sqft'] = data['Weekly_Sales'] / data['Size']
    data['hist_sales_per_sqft'] = data.groupby(['Store', 'Dept'])['sales_per_sqft'].transform(
        lambda x: x.expanding().mean().shift(1))
    
    # Refunds / Returns
    data['is_refund'] = (sales_array < 0).astype(bool)

    # STL metrics
    def decompose_stl(group, period=52):
        stl = STL(group['Weekly_Sales'], period=period, robust=True).fit()
        return stl.seasonal, stl.trend, stl.resid
    stl_results = Parallel(n_jobs=-1)(
        delayed(decompose_stl)(group)
        for _, group in data.groupby(['Store', 'Dept'])
    )
    seasonal, trend, resid = zip(*stl_results)
    data['seasonal_component'] = np.concatenate(seasonal)
    data['trend_component'] = np.concatenate(trend)
    data['residual_component'] = np.concatenate(resid)
    
    # Clean up intermediate columns
    cols_to_drop = ['running_count', 'hist_cum_sales',
                    'temp_mean', 'temp_std', 'Year', 'Month', 'Week', 'Quarter', 'total_markdown']
    data = data.drop(columns=cols_to_drop)

    # Fill NaN values
    data.fillna(0, inplace=True)
    
    return data


def calculate_mi_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # Identify categorical columns
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns

    # Convert categorical columns to 'category' dtype
    for col in categorical_columns:
        X[col] = X[col].astype("category")

    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X, y)

    # Create dataframe with results
    mi_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': mi_scores
    }).sort_values('importance', ascending=False)

    mi_importance['color'] = [
        (
            DATATYPE_COLOR_MAP["categorical"]
            if col in categorical_columns
            else DATATYPE_COLOR_MAP["numeric"]
        )
        for col in mi_importance["feature"]
    ]
    return mi_importance


def clean_feature_name(name: str) -> str:
    """
    Replace non-alphanumeric characters with underscores in a given string.

    Args:
        name (str): The string to clean.

    Returns:
        str: The cleaned string.
    """
    return re.sub(r"[^\w]+", "_", name)


def calculate_ensemble_feature_importance(
    X: pd.DataFrame, y: pd.Series, n_iterations: int = 5
) -> pd.DataFrame:
    """
    Calculate ensemble feature importance for regression using XGBoost and LightGBM.
    
    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable (continuous)
        n_iterations (int): Number of training iterations
        
    Returns:
        pd.DataFrame: Ensemble feature importance scores
    """
    models = {
        "xgboost": XGBRegressor(
            n_jobs=-1,
            enable_categorical=True
        ),
        "lightgbm": lightgbm.LGBMRegressor(
            n_jobs=-1, 
            verbose=-1
        )
    }

    feature_importance_sum = {model: pd.Series(0, index=X.columns) for model in models}
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns

    for col in categorical_columns:
        X[col] = X[col].astype("category")

    X.columns = [clean_feature_name(name) for name in X.columns]

    for i in range(n_iterations):
        X_split, _, y_split, _ = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE + i
        )

        for model_name, model in models.items():
            model.random_state = RANDOM_STATE + i

            if model_name == "xgboost":
                model.fit(X_split, y_split)
            elif model_name == "lightgbm":
                model.fit(
                    X_split, y_split, categorical_feature=categorical_columns.tolist()
                )

            importance = pd.Series(model.feature_importances_, index=X.columns)
            feature_importance_sum[model_name] += importance

    average_importance = {
        model: importance_sum / n_iterations
        for model, importance_sum in feature_importance_sum.items()
    }

    ensemble_df = pd.DataFrame(average_importance).mean(axis=1).reset_index()
    ensemble_df.columns = ["feature", "importance"]
    ensemble_df = ensemble_df.sort_values("importance", ascending=False)
    ensemble_df["color"] = [
        DATATYPE_COLOR_MAP["categorical"] if col in categorical_columns 
        else DATATYPE_COLOR_MAP["numeric"]
        for col in ensemble_df["feature"]
    ]

    return ensemble_df


def plot_feature_importances(
    df: pd.DataFrame, target_column: str, importance_method: str
) -> None:
    """
    Plot feature importances.

    Parameters:
        df (pd.DataFrame): A DataFrame with columns 'feature', 'score', and 'color'.
        target_column (str): The name of the target column.
        importance_method (str): The name of the importance method (e.g. 'Mutual Information', 'Permutation Importance').

    Returns:
        None
    """
    # Create the plot
    plt.figure(figsize=(12, 10))
    _ = sns.barplot(x="importance", y="feature", data=df, palette=df["color"].tolist())
    plt.title(f"{importance_method.title()} Scores (target: {target_column})")
    plt.xlabel(f"{importance_method.title()}")
    plt.ylabel("Features")

    # Add a legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in DATATYPE_COLOR_MAP.values()
    ]
    labels = list(DATATYPE_COLOR_MAP.keys())
    plt.legend(handles, labels, title="Data Type", loc="lower right")

    plt.tight_layout()
    plt.show()


def phik_matrix(
    df: pd.DataFrame,
    numerical_columns: list,
    target_column: str,
    feature_importances: pd.DataFrame,
) -> tuple:
    """
    Calculates the Phi_k correlation coefficient matrix for the given DataFrame and columns,
    and returns the top 10 largest phik coefficients between the target feature and other features,
    as well as the top 10 interactions between any features.

    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_columns (list): List of numerical columns.
        target_column (str): Name of the target column. Defaults to 'TARGET'.
        feature_importances (pd.DataFrame): DataFrame containing feature importances

    Returns:
        tuple: (DataFrame of top 10 target correlations, DataFrame of top 10 overall interactions)
    """
    # Calculate Phi_k correlation matrix
    corr_matrix = df.phik_matrix(interval_cols=numerical_columns)

    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(get_screen_width() / 100, 10))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Phi_k Correlation Heatmap")
    plt.show()

    # Extract correlations with the target feature
    target_correlations = corr_matrix[target_column].sort_values(ascending=False)

    # Remove self-correlation (correlation of TARGET with itself)
    target_correlations = target_correlations[
        target_correlations.index != target_column
    ]

    # Get top 10 correlations with target
    top_10_target = target_correlations.head(10)

    # Create a DataFrame with feature names and their correlations to target
    target_df = pd.DataFrame(
        {"Feature": top_10_target.index, "Phik Coefficient": top_10_target.values}
    )

    # Retrieve top interactions between features and their FI scores
    corr_df = corr_matrix.unstack().reset_index()
    corr_df.columns = ["Feature1", "Feature2", "Phik Coefficient"]
    corr_df = corr_df[corr_df["Feature1"] < corr_df["Feature2"]]
    top_interactions = corr_df[corr_df["Phik Coefficient"] > 0.5].sort_values(
        "Phik Coefficient", ascending=False
    ).head(10)

    def _get_importance(feature):
        match = feature_importances[feature_importances["feature"] == feature]
        return match["importance"].values[0] if not match.empty else 0

    top_interactions["Feature1 Score"] = top_interactions["Feature1"].apply(
        _get_importance
    )
    top_interactions["Feature2 Score"] = top_interactions["Feature2"].apply(
        _get_importance
    )

    return target_df, top_interactions


def numerical_predictor_significance_test(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    missing_strategy: str = "drop",
    sample_size: int = 10000  # Added sample size parameter
) -> Dict:
    """
    Perform Spearman correlation test with sampling for large datasets.
    """
    # Create view of only needed columns
    df_view = df[[predictor, target]]
    
    # Handle missing values efficiently
    if missing_strategy == "drop":
        df_view = df_view.dropna()
    elif missing_strategy == "median_impute":
        for col in [predictor, target]:
            median_val = df_view[col].median()
            df_view[col].fillna(median_val, inplace=True)
    else:
        raise ValueError("Invalid missing_strategy")

    # Sample data if dataset is large
    if len(df_view) > sample_size:
        df_view = df_view.sample(n=sample_size, random_state=42)

    # Convert to numpy arrays for faster computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation, p_value = stats.spearmanr(
            df_view[predictor].values, 
            df_view[target].values
        )

    results = {
        "test_name": "Spearman Correlation Test",
        "p_value": p_value,
        "correlation": correlation,
        "sample_size": len(df_view)
    }

    return results

def categorical_predictor_significance_test(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    missing_strategy: str = "drop",
    sample_size: int = 10000
) -> Dict:
    """
    Performs Kruskal-Wallis H-test with sampling for large datasets.
    """
    # Create view of only needed columns
    df_view = df[[predictor, target]]
    
    # Handle missing values efficiently
    if missing_strategy == "drop":
        df_view = df_view.dropna()
    elif missing_strategy == "most_frequent":
        mode_val = df_view[predictor].mode().iloc[0]
        df_view[predictor].fillna(mode_val, inplace=True)
    else:
        raise ValueError("Invalid missing_strategy")

    # Sample data if dataset is large
    if len(df_view) > sample_size:
        df_view = df_view.sample(n=sample_size, random_state=42)
    
    # Convert to numpy arrays and group for faster computation
    groups = [group[target].values for _, group in df_view.groupby(predictor)]
    
    # Perform Kruskal-Wallis H-test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value = stats.kruskal(*groups)

    # Calculate eta-squared effect size efficiently
    target_array = df_view[target].values
    grand_mean = np.mean(target_array)
    ss_total = np.sum((target_array - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    eta_squared = ss_between / ss_total if ss_total != 0 else 0

    results = {
        "test_name": "Kruskal-Wallis H-test",
        "p_value": p_value,
        "statistic": statistic,
        "effect_size": eta_squared,
    }

    return results

# Interpretation functions remain the same since they process only the results dictionary
def interpret_results_numerical(
    df: pd.DataFrame, results: dict, col_name: str
) -> pd.DataFrame:
    """Interpret the results of the Spearman correlation test"""
    data = {
        "Column": col_name,
        "Test Name": [results["test_name"]],
        "P-value": [round(results["p_value"], 6)],
        "Correlation": [round(results["correlation"], 4)],
        "Sample Size": [results["sample_size"]],
        "Significance": [
            "Statistically significant"
            if results["p_value"] < 0.05
            else "Not statistically significant"
        ],
        "Effect Magnitude": [],
    }

    correlation = abs(results["correlation"])
    if correlation < 0.1:
        effect_magnitude = "negligible"
    elif correlation < 0.3:
        effect_magnitude = "small"
    elif correlation < 0.5:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"

    data["Effect Magnitude"] = [effect_magnitude]
    data["Correlation Direction"] = [
        "Positive" if results["correlation"] > 0 
        else "Negative" if results["correlation"] < 0 
        else "No correlation"
    ]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    return df

def interpret_results_categorical(
    df: pd.DataFrame, results: dict, col_name: str
) -> pd.DataFrame:
    """Interpret the results of the Kruskal-Wallis test"""
    data = {
        "Column": col_name,
        "Test Name": [results["test_name"]],
        "P-value": [round(results["p_value"], 6)],
        "Test Statistic": [round(results["statistic"], 2)],
        "Effect Size (eta-squared)": [round(results["effect_size"], 4)],
        "Significance": [
            "Statistically significant"
            if results["p_value"] < 0.05
            else "Not statistically significant"
        ],
        "Effect Magnitude": [],
    }

    effect_size = results["effect_size"]
    if effect_size < 0.01:
        effect_magnitude = "negligible"
    elif effect_size < 0.06:
        effect_magnitude = "small"
    elif effect_size < 0.14:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"

    data["Effect Magnitude"] = [effect_magnitude]

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    return df


def plot_continuous_relationships(
    df: pd.DataFrame, 
    predictors: list, 
    target: str, 
    sample_size: int = 10000
) -> None:
    """
    Visualizes the relationship between multiple predictors and a continuous target as subplots.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        predictors (list): List of predictor variables.
        target (str): Name of numeric target variable.
        sample_size (int): Maximum number of samples to plot.
    """
    # Input validation
    missing_columns = [col for col in predictors + [target] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")
    
    # Sample data if larger than sample_size
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"Sampled DataFrame size: {plot_df.shape}")
    else:
        plot_df = df

    # Determine subplot grid layout
    n_predictors = len(predictors)
    ncols = 4
    nrows = (n_predictors + ncols - 1) // ncols  # Calculate rows needed

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    axes = axes.flatten()  # Flatten to easily iterate over

    # Iterate through predictors and create scatter plots
    for i, predictor in enumerate(predictors):
        sns.scatterplot(
            data=plot_df,
            x=predictor,
            y=target,
            alpha=0.6,
            ax=axes[i]
        )
        axes[i].set_title(f"{predictor} vs {target}")
        axes[i].set_xlabel(predictor)
        axes[i].set_ylabel(target)
        axes[i].grid(False)  # Remove gridlines
    
    # Hide unused subplots
    for j in range(len(predictors), len(axes)):
        axes[j].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_categorical_relationships(
    df: pd.DataFrame, 
    predictors: list,
    high_cardinality_predictors: list, 
    target: str, 
    sample_size: int = 10000
) -> None:
    """
    Visualizes the relationship between multiple categorical predictors and a continuous target,
    with special handling for high-cardinality features (`Store` and `Dept`).

    Args:
        df (pd.DataFrame): The input DataFrame.
        predictors (list): List of categorical predictor variables.
        target (str): Name of continuous target variable.
        sample_size (int): Maximum number of samples to plot.
    """
    # Input validation
    missing_columns = [col for col in predictors + [target] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")
    
    # Sample data if larger than sample_size
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"Sampled DataFrame size: {plot_df.shape}")
    else:
        plot_df = df

    # Determine subplot grid layout
    n_predictors = len(predictors)
    ncols = 2  # Base number of columns
    nrows = n_predictors + sum(1 for p in predictors if p in high_cardinality_predictors)  # Add extra rows for larger plots

    # Set up figure and gridspec for flexible subplot layout
    fig = plt.figure(figsize=(15, 3 * nrows))
    spec = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)

    current_row = 0  # Track the current row position

    for predictor in predictors:
        if predictor in ["Store", "Dept"]:
            # Allocate a row for high-cardinality features
            ax = fig.add_subplot(spec[current_row, :])  # Use full row (span 2 columns)
            sns.boxplot(
                data=plot_df,
                x=predictor,
                y=target,
                palette=COLOR_PALETTE,
                ax=ax
            )
            ax.set_title(f"{predictor} vs {target}")
            ax.set_xlabel(predictor)
            ax.set_ylabel(target)
            ax.grid(False)  # Remove gridlines
            current_row += 1
        else:
            # Allocate standard subplot (half-row)
            ax = fig.add_subplot(spec[current_row, 0])
            sns.boxplot(
                data=plot_df,
                x=predictor,
                y=target,
                palette=COLOR_PALETTE,
                ax=ax
            )
            ax.set_title(f"{predictor} vs {target}")
            ax.set_xlabel(predictor)
            ax.set_ylabel(target)
            ax.grid(False)  # Remove gridlines

            # Add barplot on the second half of the row
            ax = fig.add_subplot(spec[current_row, 1])
            sns.barplot(
                data=plot_df,
                x=predictor,
                y=target,
                palette=COLOR_PALETTE,
                estimator=np.mean,  # Show mean target value
                ax=ax
            )
            ax.set_title(f"Mean {target} by {predictor}")
            ax.set_xlabel(predictor)
            ax.set_ylabel(f"Mean {target}")
            ax.grid(False)  # Remove gridlines

            current_row += 1

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_validation_grid(df, validation_predictions, store_metrics, stores_to_plot=6):
    """
    Create a grid of validation plots for multiple stores
    """

    fig, axes = plt.subplots(round(stores_to_plot / 2), 2, figsize=(20, 25))
    axes = axes.ravel()
    
    for idx, store_id in enumerate(sorted(df['Store'].unique())[:stores_to_plot]):
        val_data = validation_predictions[store_id]
        
        # Plot actual sales
        axes[idx].plot(val_data['Date'], val_data['Actual'],
                        label='Actual Sales', color='blue', marker='o')
        
        # Plot predicted sales
        axes[idx].plot(val_data['Date'], val_data['Predicted'],
                        label='Model Forecast', color='red', linestyle='--', marker='s')
        
        # Add error bands
        axes[idx].fill_between(val_data['Date'],
                            val_data['Predicted'] * 0.9,
                            val_data['Predicted'] * 1.1,
                            color='red', alpha=0.1,
                            label='±10% Error Band')
        
        # Calculate metrics
        mape = store_metrics[store_id]['MAPE']
        rmse = store_metrics[store_id]['RMSE']
        
        # Customize subplot
        axes[idx].set_title(f'Store {store_id} - Validation Period Performance')
        axes[idx].set_xlabel('Date')
        axes[idx].set_ylabel('Weekly Sales ($)')
        axes[idx].legend()
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(False)  
        # Add metrics text
        metrics_text = f'MAPE: {mape:.2%}\nRMSE: ${rmse:,.2f}'
        axes[idx].text(0.02, 0.98, metrics_text,
                    transform=axes[idx].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Format y-axis as currency
        axes[idx].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    return fig


def plot_store_forecast_performance(store_id=None, df=None, validation_predictions=None, 
                            store_forecasts=None, store_metrics=None):
    """
    Plot full historical sales (training + validation) and future forecast sales
    If store_id is None, shows total chain performance
    
    Parameters:
    -----------
    store_id : int or None
        Store ID to plot. If None, plots total chain performance
    df_raw : pandas.DataFrame
        Raw dataframe containing full historical sales data
    validation_predictions : dict
        Dictionary containing validation predictions for each store
    store_forecasts : dict
        Dictionary containing forecasts for each store
    store_metrics : dict
        Dictionary containing validation metrics for each store
    chain_metrics : dict
        Dictionary containing total chain validation metrics
    """
    plt.figure(figsize=(15, 8))
    sns.set_style('whitegrid')
    
    # Define period boundaries
    train_end = pd.Timestamp('2012-06-10')
    validation_start = pd.Timestamp('2012-06-10')
    forecast_start = pd.Timestamp('2012-12-10')
    
    historical_data = df[df['Store'] == store_id].copy()
    title = f'Store {store_id} - Sales Forecast'
    
    # Plot training period
    train_data = historical_data[historical_data['Date'] < train_end]
    plt.plot(train_data['Date'], train_data['Weekly_Sales'],
            label='Training Data', color='gray', marker='o', alpha=0.5)
    
    # Plot validation period - actual values
    plt.plot(validation_predictions['Date'], validation_predictions['Actual'],
            label='Validation Data', color='blue', marker='o')
    
    # Plot validation period - predictions
    plt.plot(validation_predictions['Date'], validation_predictions['Predicted'],
            label='Model Fit', color='green', linestyle='--', marker='s')
    
    # Plot forecast period
    plt.plot(store_forecasts['Date'], store_forecasts['Forecast'],
            label='Future Forecast', color='red', linestyle='--', marker='s')
    
    # Add error bands for validation and forecast periods
    # Validation period bands
    plt.fill_between(validation_predictions['Date'],
                    validation_predictions['Predicted'] * 0.9,
                    validation_predictions['Predicted'] * 1.1,
                    color='green', alpha=0.1,
                    label='±10% Error Band (Historical)')
    
    # Forecast period bands
    plt.fill_between(store_forecasts['Date'],
                    store_forecasts['Forecast'] * 0.9,
                    store_forecasts['Forecast'] * 1.1,
                    color='red', alpha=0.1,
                    label='±10% Error Band (Forecast)')
    
    # Add vertical lines for period boundaries
    plt.axvline(x=validation_start, color='gray', linestyle=':',
                label='Validation Start')
    plt.axvline(x=forecast_start, color='black', linestyle=':',
                label='Forecast Start')
    
    # Customize plot
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(False)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add metrics
    mape = float(store_metrics['MAPE'])
    rmse = float(store_metrics['RMSE'])
    metrics_text = (f'Validation Metrics:\nMAPE: {mape:.2%}\n'
                   f'RMSE: ${rmse:,.2f}')
    plt.text(0.02, 0.98, metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()

def create_forecast_dataframe(df, df_stores, holiday_weeks, forecast_start, forecast_end):
    # Generate dates
    forecast_dates = pd.date_range(forecast_start, forecast_end, freq='W-FRI')

    # Create forecast df with Store, Dept, Date combinations
    forecast_df = pd.DataFrame([(store, dept, date) 
                            for store in df['Store'].unique()
                            for dept in df['Dept'].unique() 
                            for date in forecast_dates],
                            columns=['Store', 'Dept', 'Date'])
    
    # Set Holiday weeks
    forecast_df['IsHoliday'] = forecast_df['Date'].isin(pd.to_datetime(holiday_weeks))

    # Merge Type and Size to forecast
    forecast_df = forecast_df.merge(
        df_stores,
        on=['Store'],
        how='left'
    )
    forecast_df['Weekly_Sales'] = 0

    return forecast_df


def forecast_store_sales(
    df,
    model,
    model_name='xgboost',
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    validation_start=VALIDATION_START,
    validation_end=VALIDATION_END,
    forecast_start=FORECAST_START,
    forecast_end=FORECAST_END,
    decay_rate=DECAY_RATE
):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df = pd.get_dummies(df, columns=['Type'], prefix=['Type'])
    feature_cols = df.drop(columns=['Date', 'Store', 'Weekly_Sales'], axis=1).columns.to_list()
    store_forecasts = {}
    store_metrics = {}
    validation_predictions = {}
    scaler = StandardScaler()
   
    def calculate_time_weights(dates, decay_rate):
        max_year = dates.dt.year.max()
        years_old = max_year - dates.dt.year
        weights = np.maximum(1 - (years_old * decay_rate), 0)  # Prevent negative weights
        return weights / weights.mean()  # Normalize weights to mean=1
   
    for store in df['Store'].unique():
        store_data = df[df['Store'] == store].copy()
        train_data = store_data[(store_data['Date'] >= train_start) & (store_data['Date'] < train_end)]
        train_weights = calculate_time_weights(train_data['Date'], decay_rate)
        
        val_data = store_data[(store_data['Date'] >= validation_start) &
                            (store_data['Date'] <= validation_end)]
        forecast_data = store_data[(store_data['Date'] >= forecast_start) &
                            (store_data['Date'] <= forecast_end)]
        
        X_train = train_data[feature_cols]
        y_train = train_data['Weekly_Sales']
        X_val = val_data[feature_cols]
        y_val = val_data['Weekly_Sales']
        X_forecast = forecast_data[feature_cols]
        
        if X_train.empty or X_val.empty:
            continue
            
        if model_name == 'elasticnet':
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_forecast_scaled = scaler.transform(X_forecast)
            
            model.fit(X_train_scaled, y_train, sample_weight=train_weights)
            val_preds = model.predict(X_val_scaled)
            predictions = model.predict(X_forecast_scaled)
        else:
            model.fit(X_train, y_train, sample_weight=train_weights)
            val_preds = model.predict(X_val)
            predictions = model.predict(X_forecast)
            
        validation_predictions[store] = pd.DataFrame({
            'Date': val_data['Date'],
            'Actual': val_data['Weekly_Sales'],
            'Predicted': val_preds
        })
        
        store_metrics[store] = {
            'MAPE': mean_absolute_percentage_error(y_val, val_preds),
            'RMSE': np.sqrt(mean_squared_error(y_val, val_preds))
        }
        
        store_forecasts[store] = pd.DataFrame({
            'Date': forecast_data['Date'],
            'Forecast': predictions
        })
        
    all_val_data = pd.concat([v for v in validation_predictions.values()])
    chain_val_data = all_val_data.groupby('Date').agg({
        'Actual': 'sum',
        'Predicted': 'sum'
    }).reset_index()
    
    all_forecast_data = pd.concat([f for f in store_forecasts.values()])
    chain_forecast_data = all_forecast_data.groupby('Date').agg({
        'Forecast': 'sum'
    }).reset_index()

    chain_metrics = {
        'MAPE': mean_absolute_percentage_error(chain_val_data['Actual'], chain_val_data['Predicted']),
        'RMSE': np.sqrt(mean_squared_error(chain_val_data['Actual'], chain_val_data['Predicted']))
    }
    
    return {
        'store_forecasts': store_forecasts,
        'validation_predictions': validation_predictions,
        'store_metrics': store_metrics,
        'chain_metrics': chain_metrics,
        'chain_val_data': chain_val_data,
        'chain_forecast_data': chain_forecast_data
    }


def forecast_chain_sales(
    df,
    model,
    model_name='xgboost',
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    validation_start=VALIDATION_START,
    validation_end=VALIDATION_END,
    forecast_start=FORECAST_START,
    forecast_end=FORECAST_END,
    decay_rate=DECAY_RATE
):
    def calculate_time_weights(dates, decay_rate):
        max_year = dates.dt.year.max()
        years_old = max_year - dates.dt.year
        weights = np.maximum(1 - (years_old * decay_rate), 0)
        return weights / weights.mean()

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    
    df = pd.get_dummies(df, columns=['Type'], prefix=['Type'])
    feature_cols = df.drop(columns=['Date', 'Store', 'Weekly_Sales'], axis=1).columns.to_list()
    
    scaler = StandardScaler()
    
    train_data = df[(df['Date'] >= train_start) & (df['Date'] < train_end)].groupby('Date').agg({
        'Weekly_Sales': 'sum',
        **{col: 'mean' for col in feature_cols}
    }).reset_index()
    
    train_weights = calculate_time_weights(train_data['Date'], decay_rate)
    
    val_data = df[(df['Date'] >= validation_start) &
                (df['Date'] <= validation_end)].groupby('Date').agg({
        'Weekly_Sales': 'sum',
        **{col: 'mean' for col in feature_cols}
    }).reset_index()
    
    forecast_data = df[(df['Date'] >= forecast_start) &
                    (df['Date'] <= forecast_end)].groupby('Date').agg({
        'Weekly_Sales': 'sum',
        **{col: 'mean' for col in feature_cols}
    }).reset_index()
    
    X_train = train_data[feature_cols]
    y_train = train_data['Weekly_Sales']
    X_val = val_data[feature_cols]
    y_val = val_data['Weekly_Sales']
    X_forecast = forecast_data[feature_cols]
    
    if model_name == 'elasticnet':
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_forecast_scaled = scaler.transform(X_forecast)
        
        model.fit(X_train_scaled, y_train, sample_weight=train_weights)
        val_preds = model.predict(X_val_scaled)
        predictions = model.predict(X_forecast_scaled)
    else:
        model.fit(X_train, y_train, sample_weight=train_weights)
        val_preds = model.predict(X_val)
        predictions = model.predict(X_forecast)
    
    chain_data = pd.DataFrame({
        'Date': val_data['Date'],
        'Actual': val_data['Weekly_Sales'],
        'Predicted': val_preds
    })
    
    chain_metrics = {
        'MAPE': mean_absolute_percentage_error(y_val, val_preds),
        'RMSE': np.sqrt(mean_squared_error(y_val, val_preds))
    }
    
    chain_forecast = pd.DataFrame({
        'Date': forecast_data['Date'],
        'Forecast': predictions
    })
    
    return {
        'chain_forecast': chain_forecast,
        'validation_predictions': chain_data,
        'chain_metrics': chain_metrics
    }


def plot_chain_forecast_performance(df, validation_predictions, 
                                  chain_forecast, chain_metrics):
    """
    Plot chain-level historical sales and forecast performance
    """
    plt.figure(figsize=(15, 5))
    sns.set_style('whitegrid')
    
    # Define period boundaries
    train_end = pd.Timestamp('2012-06-10')
    validation_start = pd.Timestamp('2012-06-10')
    forecast_start = pd.Timestamp('2012-12-10')
    
    # Prepare chain-level historical data
    historical_data = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    
    # Plot training period
    train_data = historical_data[historical_data['Date'] < train_end]
    plt.plot(train_data['Date'], train_data['Weekly_Sales'],
            label='Training Data', color='gray', marker='o', alpha=0.5)
    
    # Plot validation period
    plt.plot(validation_predictions['Date'], validation_predictions['Actual'],
            label='Validation Data', color='blue', marker='o')
    plt.plot(validation_predictions['Date'], validation_predictions['Predicted'],
            label='Model Fit', color='green', linestyle='--', marker='s')
    
    # Plot forecast period
    plt.plot(chain_forecast['Date'], chain_forecast['Forecast'],
            label='Future Forecast', color='red', linestyle='--', marker='s')
    
    # Add error bands
    plt.fill_between(validation_predictions['Date'],
                    validation_predictions['Predicted'] * 0.9,
                    validation_predictions['Predicted'] * 1.1,
                    color='green', alpha=0.1,
                    label='±10% Error Band (Historical)')
    
    plt.fill_between(chain_forecast['Date'],
                    chain_forecast['Forecast'] * 0.9,
                    chain_forecast['Forecast'] * 1.1,
                    color='red', alpha=0.1,
                    label='±10% Error Band (Forecast)')
    
    # Add period boundaries
    plt.axvline(x=validation_start, color='gray', linestyle=':',
                label='Validation Start')
    plt.axvline(x=forecast_start, color='black', linestyle=':',
                label='Forecast Start')
    
    # Customize plot
    plt.title('Total Chain - Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(False)
    
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add metrics
    metrics_text = (f'Validation Metrics:\nMAPE: {chain_metrics["MAPE"]:.2%}\n'
                   f'RMSE: ${chain_metrics["RMSE"]:,.2f}')
    plt.text(0.02, 0.98, metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_forecasted_sales(store_sales, df_metrics):   
        # Create figure and axis for dual axes
        fig, ax1 = plt.subplots(figsize=(20, 6))
        ax2 = ax1.twinx()

        # Calculate average
        sales_avg = store_sales['Forecast'].mean()

        # Create bar plot on first axis
        bars = ax1.bar(store_sales['Store'].astype(str), store_sales['Forecast'])
        ax1.axhline(y=sales_avg, color='black', linestyle='--', label=f'Avg: ${sales_avg:,.0f}')

        # Add MAPE dots on secondary axis
        ax2.scatter(df_metrics['Store'].astype(str), df_metrics['MAPE'], color='red', label='MAPE')

        # Add data labels for bars
        for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height/1e6:.0f}M',
                        ha='center', va='bottom')

        # Customize plot
        ax1.set_title('Forecasted Total Sales and MAPE by Store')
        ax1.set_xlabel('Store ID')
        ax1.set_ylabel('Total Sales ($)')
        ax2.set_ylabel('MAPE')

        # Remove gridlines
        ax1.grid(False)
        ax2.grid(False)

        # Format x-axis
        ax1.set_xticks(range(len(store_sales)))
        ax1.set_xticklabels(store_sales['Store'], rotation=0)

        # Show both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.show()


def plot_store_metrics(df_metrics):
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))
    fig.tight_layout(pad=2.0)

    # Sort data by store ID and convert Store to int
    df_metrics_sorted = df_metrics.sort_values('Store')
    df_metrics_sorted['Store'] = df_metrics_sorted['Store'].astype(int)

    # Calculate averages
    mape_avg = df_metrics_sorted['MAPE'].mean()
    rmse_avg = df_metrics_sorted['RMSE'].mean()

    # MAPE subplot
    _ = ax1.bar(df_metrics_sorted['Store'].astype(str), df_metrics_sorted['MAPE'])
    ax1.axhline(y=mape_avg, color='red', linestyle='--', label=f'Avg: {mape_avg:.2f}%')
    ax1.set_title('MAPE by Store')
    ax1.set_xlabel(None)
    ax1.set_ylabel('MAPE (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_xticks(range(len(df_metrics_sorted)))
    ax1.set_xticklabels(df_metrics_sorted['Store'], rotation=45)

    # RMSE subplot
    _ = ax2.bar(df_metrics_sorted['Store'].astype(str), df_metrics_sorted['RMSE'])
    ax2.axhline(y=rmse_avg, color='red', linestyle='--', label=f'Avg: ${rmse_avg:.2f}')
    ax2.set_title('RMSE by Store')
    ax2.set_xlabel(None)
    ax2.set_ylabel('RMSE ($)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_xticks(range(len(df_metrics_sorted)))
    ax2.set_xticklabels(df_metrics_sorted['Store'], rotation=45)

    # Remove grids
    ax1.grid(False)
    ax2.grid(False)

    plt.show();


def train_and_plot_model_results(df, models, df_performance, trial_name):
    for model_name, model in models.items():
        print(f'Model name: {model_name}')
        start_time = time.time()
        results = forecast_store_sales(
            df,
            model,
            model_name
        )
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

        score_MAPE_store = round(results['chain_metrics']['MAPE'], 6)
        score_RMSE_store = int(results['chain_metrics']['RMSE'])
        
        df_performance.loc[df_performance['Model'] == model_name, f'MAPE_{trial_name}'] = score_MAPE_store
        df_performance.loc[df_performance['Model'] == model_name, f'RMSE_{trial_name}'] = score_RMSE_store

        plot_chain_forecast_performance(
            df,
            validation_predictions=results['chain_val_data'],
            chain_forecast=results['chain_forecast_data'],
            chain_metrics=results['chain_metrics']
        )

    return df_performance


def prepare_chain_data_for_tuning(
    df,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    validation_start=VALIDATION_START,
    validation_end=VALIDATION_END,
    decay_rate=DECAY_RATE
):
    # Create time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    
    # OneHotEncoding
    df = pd.get_dummies(df, columns=['Type'], prefix=['Type'])
    feature_cols = df.drop(columns=['Date', 'Store', 'Weekly_Sales'], axis=1).columns.to_list()
    
    # Calculate weights
    def calculate_time_weights(dates, decay_rate):
        max_year = dates.dt.year.max()
        years_old = max_year - dates.dt.year
        weights = np.maximum(1 - (years_old * decay_rate), 0)
        return weights / weights.mean()
    
    # Aggregate to chain level
    chain_data = df.groupby('Date')[['Weekly_Sales'] + feature_cols].agg({
        'Weekly_Sales': 'sum',
        **{col: 'mean' for col in feature_cols}
    }).reset_index()
    
    # Split data
    train_data = chain_data[(chain_data['Date'] >= train_start) & (chain_data['Date'] < train_end)]
    val_data = chain_data[(chain_data['Date'] >= validation_start) &
                            (chain_data['Date'] <= validation_end)]
    
    # Calculate weights for training data
    train_weights = calculate_time_weights(train_data['Date'], decay_rate)
    
    X_train = train_data[feature_cols]
    y_train = train_data['Weekly_Sales']
    X_val = val_data[feature_cols]
    y_val = val_data['Weekly_Sales']
    
    return X_train, y_train, X_val, y_val, feature_cols, train_weights


def plot_forecasted_sales_change(df_growth, df_metrics):   
        # Create figure and axis for dual axes
        _, ax1 = plt.subplots(figsize=(20, 6))
        ax2 = ax1.twinx()

        # Calculate average
        growth_avg = df_growth['Change'].mean()

        # Create bar plot on first axis
        bars = ax1.bar(df_growth['Store'].astype(str), df_growth['Change'])
        ax1.axhline(y=growth_avg, color='black', linestyle='--', label=f'Avg: {growth_avg:.1%}')

        # Add MAPE dots on secondary axis
        ax2.scatter(df_metrics['Store'].astype(str), df_metrics['MAPE'], color='red', label='MAPE')

        # Add data labels for bars
        for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom')

        # Customize plot
        ax1.set_title('Forecasted Total Sales Change and MAPE by Store')
        ax1.set_xlabel('Store ID')
        ax1.set_ylabel('Forecasted Sales Change %')
        ax2.set_ylabel('MAPE')

        # Remove gridlines
        ax1.grid(False)
        ax2.grid(False)

        # Format x-axis
        ax1.set_xticks(range(len(df_growth)))
        ax1.set_xticklabels(df_growth['Store'], rotation=0)

        # Show both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.show()