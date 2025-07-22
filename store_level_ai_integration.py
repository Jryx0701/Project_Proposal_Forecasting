import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Theta, ExponentialSmoothing
from darts.metrics import mape
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def train_and_forecast_product_store_level(df,warehouse, product_code, forecast_days=30):
    """
    Trains an XGBoost model for a specific product and forecasts future sales.
    
    Args:
        df (pd.DataFrame): Full dataset with columns ['TrnDate', 'Product', 'IssueQuantity'].
        product_code (str): The product code to train on (e.g., 'GMS-000013').
        forecast_days (int): Number of days to forecast into the future.
    
    Returns:
        dict: {
            'product': str,
            'rmse': float,
            'mean_sales': float,
            'rmse_percent': float,
            'forecast_df': pd.DataFrame
        }
    """

    # 1. Filter product
    df = df[df['Warehouse'] == warehouse].copy()
    df_product = df[df['PName'] == product_code].copy()

    if df_product.empty:
        return {'error': f"No data found for product {product_code}"}

    # 2. Preprocess
    df_product = df_product.groupby('TrnDate', as_index=False)['IssueQuantity'].sum()
    df_product = df_product.sort_values(by='TrnDate')

    # 3. Remove outliers
    Q1 = df_product['IssueQuantity'].quantile(0.25)
    Q3 = df_product['IssueQuantity'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_product = df_product[(df_product['IssueQuantity'] >= lower) & (df_product['IssueQuantity'] <= upper)]

    # 4. Lag features
    for lag in [1, 2, 3]:
        df_product[f'lag_{lag}'] = df_product['IssueQuantity'].shift(lag)

    df_lagged = df_product.dropna().copy()

    if len(df_lagged) < 10:
        return {'error': f"Not enough data to train model for {product_code}"}

    # 5. Define features
    X = df_lagged[['lag_1', 'lag_2', 'lag_3']]
    y = df_lagged['IssueQuantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mean_actual = y_test.mean()
    rmse_percent = (rmse / mean_actual) * 100

    # 6. Forecast future days
    future_preds = []
    last_known = df_lagged.iloc[-1][['lag_1', 'lag_2', 'lag_3']].tolist()
    future_dates = pd.date_range(start=df_lagged['TrnDate'].max() + pd.Timedelta(days=1), periods=forecast_days)

    for _ in range(forecast_days):
        next_input = np.array(last_known).reshape(1, -1)
        pred = model.predict(next_input)[0]
        future_preds.append(pred)
        last_known = [pred] + last_known[:2]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_IssueQuantity': future_preds
    })

    return {
        'product': product_code,
        'rmse': round(rmse, 2),
        'mean_sales': round(mean_actual, 2),
        'rmse_percent': round(rmse_percent, 2),
        'forecast_df': forecast_df
    }

def get_top_products_store(df,warehouse, top_n=10):
    """
    Returns the top N products based on total IssueQuantity.

    Args:
        df (pd.DataFrame): The input DataFrame with columns ['Product', 'IssueQuantity'].
        top_n (int): Number of top products to return.

    Returns:
        pd.DataFrame: DataFrame with columns ['Product', 'TotalSales'] sorted descending.
    """
    
    df = df[df['Warehouse'] == warehouse].copy()
    
    top_products = (
        df.groupby('PName')['IssueQuantity']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'IssueQuantity': 'TotalSales'})
        .head(top_n)
    )
    return top_products
