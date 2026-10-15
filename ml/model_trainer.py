import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from database.db import get_db_connection
from ml.preprocessor import FoodDemandPreprocessor

def train_and_evaluate_models(csv_path=None, test_ratio=0.20):
    """
    Trains Random Forest Regressor, Linear Regression, and Gradient Boosting Regressor
    using chronological split. Evaluates MAE, RMSE, MAPE, R2, records to DB, and persists artifacts.
    """
    if csv_path is None:
        csv_path = os.path.join(Config.DATA_DIR, 'historical_food_waste_data.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading dataset from: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    # 1. Validation & Preprocessing
    preprocessor = FoodDemandPreprocessor()
    is_valid, issues, clean_df = preprocessor.validate_csv(raw_df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {issues}")

    engineered_df = preprocessor.engineer_features(clean_df, is_training=True)

    # 2. Strict Chronological Train/Test Split
    engineered_df = engineered_df.sort_values(by='Date').reset_index(drop=True)
    split_index = int(len(engineered_df) * (1 - test_ratio))

    train_df = engineered_df.iloc[:split_index]
    test_df = engineered_df.iloc[split_index:]

    X_train = train_df[preprocessor.feature_columns]
    y_train = train_df[preprocessor.target_column]

    X_test = test_df[preprocessor.feature_columns]
    y_test = test_df[preprocessor.target_column]

    print(f"Chronological split completed:")
    print(f"  Training samples: {len(X_train)} (Dates: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  Testing samples:  {len(X_test)} (Dates: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")

    # 3. Model Definitions
    models = {
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        'Linear Regression (Baseline)': LinearRegression(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
    }

    results = {}
    best_model_name = None
    lowest_mae = float('inf')

    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Enforce non-negative demand constraint
        y_pred = np.maximum(0, y_pred)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(mean_absolute_percentage_error(y_test, y_pred)) * 100.0  # Percentage
        r2 = float(r2_score(y_test, y_pred))

        # Feature importances if supported
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = {
                feat: round(float(imp), 4)
                for feat, imp in sorted(zip(preprocessor.feature_columns, importances), key=lambda x: x[1], reverse=True)
            }
        elif hasattr(model, 'coef_'):
            coefs = np.abs(model.coef_)
            feature_importance = {
                feat: round(float(c), 4)
                for feat, c in sorted(zip(preprocessor.feature_columns, coefs), key=lambda x: x[1], reverse=True)
            }

        results[name] = {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2),
            'r2': round(r2, 4),
            'sample_size': len(X_test),
            'feature_importance': feature_importance,
            'model_object': model
        }

        print(f"  MAE:  {mae:.2f} portions")
        print(f"  RMSE: {rmse:.2f} portions")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")

        if mae < lowest_mae:
            lowest_mae = mae
            best_model_name = name

    print(f"\n>>> Best Performing Model: {best_model_name} (Lowest MAE: {lowest_mae:.2f})")

    # 4. Save artifacts
    # Save best model
    best_model_path = os.path.join(Config.MODEL_DIR, 'best_model.joblib')
    joblib.dump(results[best_model_name]['model_object'], best_model_path)

    # Save all models for live switching/comparison
    for name, res in results.items():
        clean_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        m_path = os.path.join(Config.MODEL_DIR, f"{clean_name}.joblib")
        joblib.dump(res['model_object'], m_path)

    # Save preprocessor
    preprocessor_path = os.path.join(Config.MODEL_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)

    # Save evaluation summary JSON
    summary_data = {
        'best_model': best_model_name,
        'models': {
            k: {
                'mae': v['mae'],
                'rmse': v['rmse'],
                'mape': v['mape'],
                'r2': v['r2'],
                'sample_size': v['sample_size'],
                'feature_importance': v['feature_importance']
            }
            for k, v in results.items()
        }
    }
    with open(os.path.join(Config.MODEL_DIR, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)

    # 5. Persist to SQLite model_evaluation_results
    conn = get_db_connection()
    cur = conn.cursor()
    # Clear prior evaluation logs to avoid clutter
    cur.execute("DELETE FROM model_evaluation_results")

    for name, res in results.items():
        is_best = 1 if name == best_model_name else 0
        cur.execute("""
            INSERT INTO model_evaluation_results 
            (model_name, mae, rmse, mape, r2_score, sample_size, best_model_flag, feature_importance_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name, res['mae'], res['rmse'], res['mape'], res['r2'], 
            res['sample_size'], is_best, json.dumps(res['feature_importance'])
        ))
    conn.commit()
    conn.close()

    print("Model evaluation metrics successfully saved to database and model directory.")
    return summary_data

if __name__ == '__main__':
    train_and_evaluate_models()
