import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score , mean_squared_error
from sklearn.model_selection import train_test_split
from random import sample
from datetime import timedelta
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


def train_linear_regression(pdf,test_size=0.2):
    
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

    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model,"Linear_Regression_tablets.pkl")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results = {
        "model": model,
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "coefficients": model.coef_,
        "intercept": model.intercept_
    }

    return results
    
    
def forecast_product(pdf, days_ahead=7):

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

        pct_change_1 = 0
        pct_change_3 = 0

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
            pct_change_1,
            pct_change_3,
            pdf['ram_normalized'].iloc[-1],
            pdf['storage_normalized'].iloc[-1],
            pdf['specs_score'].iloc[-1]
        ]

        pred = model.predict([row])[0]

        forecasts.append(pred)
        history_prices.append(pred)

    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]

    return {
        "forecast_dates": forecast_dates,
        "forecast_prices": forecasts,
        "model_type": "Multiple Linear Regression"
    }

def evaluate_model_on_all_products(filepath, min_obs=10):

    df = load_and_preprocess_data(filepath)

    results = []

    for product_key in df['product_key'].unique():

        pdf = df[df['product_key'] == product_key]

        if len(pdf) < min_obs:
            continue

        try:
            metrics = train_linear_regression(pdf)

            results.append({
                "product": product_key,
                "n_obs": len(pdf),
                "train_mae": metrics["train_mae"],
                "test_mae": metrics["test_mae"],
                "train_r2": metrics["train_r2"],
                "test_r2": metrics["test_r2"],
                "train_rmse": metrics["train_rmse"],
                "test_rmse": metrics["test_rmse"]
            })

        except:
            continue

    results_df = pd.DataFrame(results)

    return {
        "model_name": "Multiple Linear Regression",
        "n_products": len(results_df),
        "avg_train_mae": results_df["train_mae"].mean(),
        "avg_test_mae": results_df["test_mae"].mean(),
        "avg_train_r2": results_df["train_r2"].mean(),
        "avg_test_r2": results_df["test_r2"].mean(),
        "avg_train_rmse": results_df["train_rmse"].mean(),
        "avg_test_rmse": results_df["test_rmse"].mean(),
        "std_test_mae": results_df["test_mae"].std(),
        "std_test_r2": results_df["test_r2"].std(),
        "details": results_df
    }
    
    
if __name__ == "__main__":

    print("="*80)
    print("🚀 MULTIPLE LINEAR REGRESSION MODEL EVALUATION")
    print("="*80)

    filepath='tablets_cleaned_continuous.csv'

    print("\n📊 Evaluating model on all products...")

    summary = evaluate_model_on_all_products(filepath)

    print("\n" + "="*80)
    print("📈 RESULTS")
    print("="*80)

    print(f"\nProducts tested: {summary['n_products']}")

    print("\nTRAINING PERFORMANCE")
    print(f"MAE:  {summary['avg_train_mae']:,.2f}")
    print(f"R²:   {summary['avg_train_r2']:.4f}")
    print(f"RMSE: {summary['avg_train_rmse']:,.2f}")

    print("\nTEST PERFORMANCE")
    print(f"MAE:  {summary['avg_test_mae']:,.2f} ± {summary['std_test_mae']:,.2f}")
    print(f"R²:   {summary['avg_test_r2']:.4f} ± {summary['std_test_r2']:.4f}")
    print(f"RMSE: {summary['avg_test_rmse']:,.2f}")


    print("\n✅ Evaluation Complete")        





