import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath='tablets_cleaned_continuous.csv'):
    df = pd.read_csv(filepath)

    df['price'] = df['price'].astype(str)
    df['price'] = df['price'].str.replace('EGP', '', regex=False)
    df['price'] = df['price'].str.replace(',', '', regex=False)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])

    df['product_key'] = (
        df['name'].str.lower().str.strip() + ' ' +
        df['website'].str.lower() + ' ' +
        df['ram_gb'].astype(str) + ' ' +
        df['storage_gb'].astype(str)
    )

    df_daily = df.groupby(['product_key', 'date']).agg({
        'price': 'mean',
        'name': 'first',
        'brand': 'first',
        'website': 'first',
        'ram_gb': 'first',
        'storage_gb': 'first',
        'URL': 'last',
        'timestamp': 'first'
    }).reset_index()

    df_daily = df_daily.sort_values(['product_key', 'date'])

    return df_daily


def engineer_features(pdf):
    pdf = pdf.sort_values('date').copy()

    pdf['day_index'] = (pdf['date'] - pdf['date'].min()).dt.days
    pdf['dayofweek'] = pdf['date'].dt.dayofweek
    pdf['day_of_month'] = pdf['date'].dt.day
    pdf['month'] = pdf['date'].dt.month

    pdf['rolling_avg_3'] = pdf['price'].rolling(3, min_periods=1).mean()
    pdf['rolling_avg_7'] = pdf['price'].rolling(7, min_periods=1).mean()
    pdf['rolling_std_3'] = pdf['price'].rolling(3, min_periods=1).std().fillna(0)

    pdf['price_lag_1'] = pdf['price'].shift(1).fillna(pdf['price'].iloc[0])
    pdf['price_lag_3'] = pdf['price'].shift(3).fillna(pdf['price'].iloc[0])
    pdf['price_lag_7'] = pdf['price'].shift(7).fillna(pdf['price'].iloc[0])

    pdf['pct_change_1'] = pdf['price'].pct_change().fillna(0)
    pdf['pct_change_3'] = pdf['price'].pct_change(3).fillna(0)

    pdf['ram_normalized'] = pdf['ram_gb'] / 16.0
    pdf['storage_normalized'] = pdf['storage_gb'] / 1024.0
    pdf['specs_score'] = (pdf['ram_gb'] / 4.0) + (pdf['storage_gb'] / 128.0)

    return pdf


def forecast_product(pdf, days_ahead=7):
    """
    ✅ FIXED - Returns all required fields for Streamlit
    """
    pdf = engineer_features(pdf)
    feature_cols = [
        'day_index', 'dayofweek', 'day_of_month', 'month',
        'rolling_avg_3', 'rolling_avg_7', 'rolling_std_3',
        'price_lag_1', 'price_lag_3', 'price_lag_7',
        'pct_change_1', 'pct_change_3',
        'ram_normalized', 'storage_normalized', 'specs_score'
    ]

    X = pdf[feature_cols]
    y = pdf['price']

    model = LinearRegression()
    model.fit(X, y)

    history_prices = list(pdf['price'].values)
    last_date = pdf['date'].iloc[-1]
    last_day_index = pdf['day_index'].iloc[-1]

    forecasts = []

    for i in range(days_ahead):
        future_date = last_date + timedelta(days=i+1)

        price_lag_1 = history_prices[-1]
        price_lag_3 = history_prices[-3] if len(history_prices) >= 3 else history_prices[0]
        price_lag_7 = history_prices[-7] if len(history_prices) >= 7 else history_prices[0]

        rolling_avg_3 = np.mean(history_prices[-3:])
        rolling_avg_7 = np.mean(history_prices[-7:])
        rolling_std_3 = np.std(history_prices[-3:])

        row = [
            last_day_index+i+1,
            future_date.dayofweek,
            future_date.day,
            future_date.month,
            rolling_avg_3,
            rolling_avg_7,
            rolling_std_3,
            price_lag_1,
            price_lag_3,
            price_lag_7,
            0,  # pct_change_1
            0,  # pct_change_3
            pdf['ram_normalized'].iloc[-1],
            pdf['storage_normalized'].iloc[-1],
            pdf['specs_score'].iloc[-1]
        ]

        pred = model.predict([row])[0]
        forecasts.append(pred)
        history_prices.append(pred)

    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    
    # Calculate metrics
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Confidence level
    n = len(pdf)
    if n >= 30:
        confidence = "High"
    elif n >= 15:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ✅ Return ALL required fields
    return {
        'pdf': pdf,
        'forecast_dates': forecast_dates,
        'forecast_prices': np.array(forecasts),
        'mae': mae,
        'r2': r2,
        'last_price': float(pdf['price'].iloc[-1]),
        'avg_price': float(pdf['price'].mean()),
        'min_price': float(pdf['price'].min()),
        'max_price': float(pdf['price'].max()),
        'n_obs': n,
        'confidence': confidence,
        'model_type': 'Multiple Linear Regression'
    }
