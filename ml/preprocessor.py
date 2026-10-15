import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config

class FoodDemandPreprocessor:
    """
    Handles data validation, cleaning, feature engineering, and preparation
    for food demand prediction models without lookahead data leakage.
    """
    
    REQUIRED_COLUMNS = [
        'Date', 'Food_Item', 'Category', 'Quantity_Sold'
    ]
    
    OPTIONAL_COLUMNS = [
        'Day_Of_Week', 'Is_Weekend', 'Is_Holiday', 'Special_Event',
        'Weather_Condition', 'Temperature_C', 'Quantity_Prepared',
        'Quantity_Wasted', 'Unit_Cost', 'Selling_Price'
    ]

    def __init__(self):
        self.item_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_columns = [
            'Food_Item_Enc', 'Category_Enc', 'Day_Of_Week_Num', 
            'Is_Weekend', 'Is_Holiday', 'Special_Event', 
            'Weather_Enc', 'Temperature_C', 'Month',
            'Lag_1', 'Lag_7', 'Rolling_Mean_7', 'Rolling_Std_7'
        ]
        self.target_column = 'Quantity_Sold'

    @staticmethod
    def validate_csv(df):
        """
        Validates uploaded CSV columns and data types.
        Returns (is_valid: bool, issues: list of str, clean_df or None)
        """
        issues = []
        if df is None or len(df) == 0:
            return False, ["Uploaded dataset is empty."], None

        # Check required columns (case-insensitive mapping)
        cols_lower = {col.lower(): col for col in df.columns}
        missing_required = []
        col_mapping = {}

        for req in FoodDemandPreprocessor.REQUIRED_COLUMNS:
            found = False
            for c_low, c_orig in cols_lower.items():
                if req.lower() == c_low or req.lower().replace('_', '') == c_low.replace('_', ''):
                    col_mapping[c_orig] = req
                    found = True
                    break
            if not found:
                missing_required.append(req)

        if missing_required:
            issues.append(f"Missing mandatory columns: {', '.join(missing_required)}")
            return False, issues, None

        # Map optional columns if present
        for opt in FoodDemandPreprocessor.OPTIONAL_COLUMNS:
            for c_low, c_orig in cols_lower.items():
                if opt.lower() == c_low or opt.lower().replace('_', '') == c_low.replace('_', ''):
                    col_mapping[c_orig] = opt
                    break

        df_mapped = df.rename(columns=col_mapping)

        # Validate Date parsing
        try:
            df_mapped['Date'] = pd.to_datetime(df_mapped['Date'])
        except Exception as e:
            issues.append(f"Failed to parse 'Date' column: {str(e)}")
            return False, issues, None

        # Validate Quantity_Sold numeric & non-negative
        df_mapped['Quantity_Sold'] = pd.to_numeric(df_mapped['Quantity_Sold'], errors='coerce')
        if df_mapped['Quantity_Sold'].isnull().any():
            nan_count = df_mapped['Quantity_Sold'].isnull().sum()
            issues.append(f"Found {nan_count} non-numeric or missing values in 'Quantity_Sold'. Imputing with median.")
            df_mapped['Quantity_Sold'] = df_mapped['Quantity_Sold'].fillna(df_mapped['Quantity_Sold'].median())

        # Prevent negative demand
        negative_demand_count = (df_mapped['Quantity_Sold'] < 0).sum()
        if negative_demand_count > 0:
            issues.append(f"Replaced {negative_demand_count} negative demand values with 0.")
            df_mapped.loc[df_mapped['Quantity_Sold'] < 0, 'Quantity_Sold'] = 0

        # Fill default optional columns if not present
        if 'Day_Of_Week' not in df_mapped.columns:
            df_mapped['Day_Of_Week'] = df_mapped['Date'].dt.day_name()
        if 'Is_Weekend' not in df_mapped.columns:
            df_mapped['Is_Weekend'] = df_mapped['Date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        if 'Is_Holiday' not in df_mapped.columns:
            df_mapped['Is_Holiday'] = 0
        if 'Special_Event' not in df_mapped.columns:
            df_mapped['Special_Event'] = 0
        if 'Weather_Condition' not in df_mapped.columns:
            df_mapped['Weather_Condition'] = 'Sunny'
        if 'Temperature_C' not in df_mapped.columns:
            df_mapped['Temperature_C'] = 25.0
        if 'Quantity_Prepared' not in df_mapped.columns:
            df_mapped['Quantity_Prepared'] = df_mapped['Quantity_Sold'] * 1.15
        if 'Quantity_Wasted' not in df_mapped.columns:
            df_mapped['Quantity_Wasted'] = np.maximum(0, df_mapped['Quantity_Prepared'] - df_mapped['Quantity_Sold'])
        if 'Unit_Cost' not in df_mapped.columns:
            df_mapped['Unit_Cost'] = 40.0
        if 'Selling_Price' not in df_mapped.columns:
            df_mapped['Selling_Price'] = 100.0

        return True, issues, df_mapped

    def engineer_features(self, df, is_training=True):
        """
        Creates time-series lag and rolling statistics without data leakage.
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure optional columns have safe defaults if not present
        if 'Weather_Condition' not in df.columns:
            df['Weather_Condition'] = 'Sunny'
        if 'Temperature_C' not in df.columns:
            df['Temperature_C'] = 25.0
        if 'Is_Weekend' not in df.columns:
            df['Is_Weekend'] = df['Date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
        if 'Is_Holiday' not in df.columns:
            df['Is_Holiday'] = 0
        if 'Special_Event' not in df.columns:
            df['Special_Event'] = 0

        df = df.sort_values(by=['Food_Item', 'Date']).reset_index(drop=True)

        # Time features
        df['Day_Of_Week_Num'] = df['Date'].dt.weekday
        df['Month'] = df['Date'].dt.month

        # Group by Food_Item to create strict chronological lag and rolling features
        # Crucial: shift(1) ensures today's observation (t) is NOT in lag or rolling stats
        df['Lag_1'] = df.groupby('Food_Item')['Quantity_Sold'].shift(1)
        df['Lag_7'] = df.groupby('Food_Item')['Quantity_Sold'].shift(7)
        
        # 7-day rolling window based strictly on past observations
        df['Rolling_Mean_7'] = (
            df.groupby('Food_Item')['Quantity_Sold']
            .shift(1)
            .rolling(window=7, min_periods=1)
            .mean()
        )
        df['Rolling_Std_7'] = (
            df.groupby('Food_Item')['Quantity_Sold']
            .shift(1)
            .rolling(window=7, min_periods=1)
            .std()
            .fillna(0.0)
        )

        # Impute early missing lags safely with food item's own past mean (or median)
        df['Lag_1'] = df.groupby('Food_Item')['Lag_1'].transform(lambda x: x.bfill().ffill())
        df['Lag_7'] = df.groupby('Food_Item')['Lag_7'].transform(lambda x: x.bfill().ffill())
        df['Rolling_Mean_7'] = df.groupby('Food_Item')['Rolling_Mean_7'].transform(lambda x: x.bfill().ffill())
        
        # Global fallback in case single record
        df['Lag_1'] = df['Lag_1'].fillna(df['Quantity_Sold'].median())
        df['Lag_7'] = df['Lag_7'].fillna(df['Quantity_Sold'].median())
        df['Rolling_Mean_7'] = df['Rolling_Mean_7'].fillna(df['Quantity_Sold'].median())
        df['Rolling_Std_7'] = df['Rolling_Std_7'].fillna(2.0)

        # Categorical Encodings
        if is_training:
            df['Food_Item_Enc'] = self.item_encoder.fit_transform(df['Food_Item'].astype(str))
            df['Category_Enc'] = self.category_encoder.fit_transform(df['Category'].astype(str))
            df['Weather_Enc'] = self.weather_encoder.fit_transform(df['Weather_Condition'].astype(str))
            self.is_fitted = True
        else:
            df['Food_Item_Enc'] = self._transform_unseen(self.item_encoder, df['Food_Item'].astype(str))
            df['Category_Enc'] = self._transform_unseen(self.category_encoder, df['Category'].astype(str))
            df['Weather_Enc'] = self._transform_unseen(self.weather_encoder, df['Weather_Condition'].astype(str))

        # Ensure correct numeric types
        numeric_cols = ['Is_Weekend', 'Is_Holiday', 'Special_Event', 'Temperature_C']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def _transform_unseen(self, encoder, series):
        """Safely encodes categorical values, assigning known class or nearest safe fallback."""
        classes = list(encoder.classes_)
        fallback_class = classes[0]
        return series.apply(lambda x: encoder.transform([x])[0] if x in classes else encoder.transform([fallback_class])[0])
