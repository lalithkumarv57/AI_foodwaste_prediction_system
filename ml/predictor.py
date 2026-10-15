import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from database.db import get_db_connection, query_db, execute_db

class DemandPredictor:
    """
    ML inference engine for predicting future food demand, estimating waste risk,
    and generating safe, sustainable preparation recommendations.
    """

    def __init__(self):
        self.preprocessor_path = os.path.join(Config.MODEL_DIR, 'preprocessor.joblib')
        self.best_model_path = os.path.join(Config.MODEL_DIR, 'best_model.joblib')
        self.preprocessor = None
        self.models = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Loads saved models and preprocessor from disk."""
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = joblib.load(self.preprocessor_path)

        # Load best model
        if os.path.exists(self.best_model_path):
            self.models['best'] = joblib.load(self.best_model_path)

        # Load all specific models if present
        model_names = [
            'random_forest_regressor',
            'linear_regression_baseline',
            'gradient_boosting_regressor'
        ]
        for name in model_names:
            p = os.path.join(Config.MODEL_DIR, f"{name}.joblib")
            if os.path.exists(p):
                self.models[name] = joblib.load(p)

    def get_historical_item_stats(self, food_item_id, target_date_str):
        """
        Retrieves real chronological historical sales from SQLite to calculate
        lag_1, lag_7, and rolling statistics up to the target date.
        """
        conn = get_db_connection()
        cur = conn.cursor()

        # Query recent sales prior to target_date
        cur.execute("""
            SELECT date, quantity_sold 
            FROM daily_sales 
            WHERE food_item_id = ? AND date < ? 
            ORDER BY date DESC 
            LIMIT 30
        """, (food_item_id, target_date_str))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            # Fallback if no prior sales logged
            return {
                'lag_1': 50.0,
                'lag_7': 50.0,
                'rolling_mean_7': 50.0,
                'rolling_std_7': 5.0
            }

        sales_series = [r['quantity_sold'] for r in rows]
        lag_1 = float(sales_series[0]) if len(sales_series) >= 1 else 50.0
        lag_7 = float(sales_series[6]) if len(sales_series) >= 7 else lag_1
        
        last_7 = sales_series[:7]
        rolling_mean_7 = float(np.mean(last_7))
        rolling_std_7 = float(np.std(last_7)) if len(last_7) > 1 else 3.0

        return {
            'lag_1': lag_1,
            'lag_7': lag_7,
            'rolling_mean_7': round(rolling_mean_7, 1),
            'rolling_std_7': round(rolling_std_7, 1)
        }

    def predict_demand(self, food_item_id, target_date_str, model_choice='best', 
                       weather='Sunny', temp_c=25.0, is_holiday=0, special_event=0):
        """
        Runs ML model inference for the given food item and date.
        """
        if self.preprocessor is None or not self.models:
            self.load_artifacts()
            if not self.models:
                raise RuntimeError("Models are not trained yet. Run model training first.")

        # Fetch item details
        item = query_db("SELECT * FROM food_items WHERE id = ?", (food_item_id,), one=True)
        if not item:
            raise ValueError(f"Food item with ID {food_item_id} not found.")

        # Compute calendar features
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        day_of_week_num = target_date.weekday()
        is_weekend = 1 if day_of_week_num >= 5 else 0
        month = target_date.month

        # Retrieve true historical lag statistics
        stats = self.get_historical_item_stats(food_item_id, target_date_str)

        # Categorical encodings
        item_enc = self.preprocessor._transform_unseen(self.preprocessor.item_encoder, pd.Series([item['name']]))[0]
        cat_enc = self.preprocessor._transform_unseen(self.preprocessor.category_encoder, pd.Series([item['category']]))[0]
        weather_enc = self.preprocessor._transform_unseen(self.preprocessor.weather_encoder, pd.Series([weather]))[0]

        feature_vector = pd.DataFrame([{
            'Food_Item_Enc': item_enc,
            'Category_Enc': cat_enc,
            'Day_Of_Week_Num': day_of_week_num,
            'Is_Weekend': is_weekend,
            'Is_Holiday': int(is_holiday),
            'Special_Event': int(special_event),
            'Weather_Enc': weather_enc,
            'Temperature_C': float(temp_c),
            'Month': month,
            'Lag_1': stats['lag_1'],
            'Lag_7': stats['lag_7'],
            'Rolling_Mean_7': stats['rolling_mean_7'],
            'Rolling_Std_7': stats['rolling_std_7']
        }])[self.preprocessor.feature_columns]

        # Select model
        model = self.models.get(model_choice)
        if model is None:
            # Fallback to best model
            model = self.models.get('best')

        pred_val = float(model.predict(feature_vector)[0])
        # Non-negative demand constraint
        pred_val = max(0.0, round(pred_val, 1))

        return {
            'food_item_id': food_item_id,
            'food_item_name': item['name'],
            'category': item['category'],
            'unit': item['unit'],
            'unit_cost': item['unit_cost'],
            'selling_price': item['selling_price'],
            'target_date': target_date_str,
            'predicted_demand': pred_val,
            'lag_1': stats['lag_1'],
            'lag_7': stats['lag_7'],
            'rolling_mean_7': stats['rolling_mean_7'],
            'rolling_std_7': stats['rolling_std_7'],
            'model_used': model_choice
        }

    @staticmethod
    def calculate_recommendation(predicted_demand, usable_inventory=0.0, 
                                 safety_buffer_pct=10.0, demand_std=None):
        """
        Computes recommended preparation quantity using the formula:
        Recommended Preparation Quantity = MAX(0, Predicted Demand + Safety Stock - Usable Inventory)
        
        Safety Stock is dynamically configurable based on buffer % or standard deviation.
        """
        usable_inventory = max(0.0, float(usable_inventory))
        safety_buffer_pct = max(0.0, float(safety_buffer_pct))

        # Compute Safety Stock
        if demand_std is not None and demand_std > 0:
            # Z = 1.65 for a 95% service level
            safety_stock = round(1.65 * float(demand_std), 1)
        else:
            safety_stock = round(predicted_demand * (safety_buffer_pct / 100.0), 1)

        # Recommended Prep Formula
        recommended_prep = max(0.0, round(predicted_demand + safety_stock - usable_inventory, 1))

        return {
            'predicted_demand': predicted_demand,
            'safety_stock': safety_stock,
            'safety_buffer_pct': safety_buffer_pct,
            'usable_inventory': usable_inventory,
            'recommended_prep': recommended_prep
        }

    @staticmethod
    def analyze_waste_risk(planned_preparation, predicted_demand, usable_inventory=0.0, unit_cost=40.0):
        """
        Estimates the risk of excess food using the difference between planned quantity and expected demand.
        Clearly distinguishes:
        - Predicted Demand
        - Usable Inventory
        - Planned Preparation
        - Estimated Excess / Waste
        """
        planned_prep = max(0.0, float(planned_preparation))
        total_supply = planned_prep + usable_inventory
        excess = max(0.0, round(total_supply - predicted_demand, 1))

        excess_ratio = (excess / planned_prep) if planned_prep > 0 else 0.0

        if excess_ratio > 0.20:
            risk_level = "High"
            risk_badge = "danger"
            risk_description = "Significant overproduction risk. Excess food is likely to exceed acceptable holding thresholds."
        elif excess_ratio >= 0.08:
            risk_level = "Medium"
            risk_badge = "warning"
            risk_description = "Moderate buffer. Adequate for sudden rushes, but monitor shelf-life closely."
        else:
            risk_level = "Low"
            risk_badge = "success"
            risk_description = "Optimal lean preparation. Closely aligned with predicted demand to minimize waste."

        estimated_waste_cost = round(excess * unit_cost, 2)
        
        # Benchmark comparison: typical kitchen wastes ~18% overproduction
        baseline_waste = round(planned_prep * 0.18, 1)
        potential_cost_savings = round(max(0.0, (baseline_waste - excess) * unit_cost), 2)

        return {
            'planned_preparation': planned_prep,
            'predicted_demand': predicted_demand,
            'estimated_waste_portions': excess,
            'excess_percentage': round(excess_ratio * 100.0, 1),
            'risk_level': risk_level,
            'risk_badge': risk_badge,
            'risk_description': risk_description,
            'estimated_waste_cost': estimated_waste_cost,
            'potential_cost_savings': potential_cost_savings
        }

    def generate_actionable_recommendations(self, item_name, predicted_demand, 
                                            recommended_prep, usable_inventory,
                                            waste_risk, day_of_week, is_weekend, is_holiday):
        """
        Generates contextual, evidence-based recommendations supporting SDG 12.
        """
        tips = []

        if usable_inventory > 0:
            tips.append(f"Usable inventory ({usable_inventory} portions) was factored in. Preparing only {recommended_prep} portions avoids duplicate inventory decay.")

        if waste_risk['risk_level'] == 'High':
            tips.append(f"High Waste Alert: Planned prep exceeds predicted consumption by {waste_risk['excess_percentage']}%. Cut prep batch size by at least {waste_risk['estimated_waste_portions']} portions.")
        elif waste_risk['risk_level'] == 'Low':
            tips.append(f"Lean Production: Current planned quantity matches demand closely. Safety buffer protects against customer stockouts.")

        if is_weekend:
            tips.append(f"Weekend pattern: Customer demand fluctuates more widely on {day_of_week}. Maintain a dynamic safety buffer.")
        elif is_holiday:
            tips.append(f"Holiday condition: Re-check meal-rush hours; adjust cold/hot station holding limits.")

        return tips

    def record_prediction(self, food_item_id, target_date, model_used,
                          predicted_demand, recommended_prep, safety_stock,
                          usable_inventory, risk_level, estimated_waste):
        """
        Persists prediction details into demand_predictions table.
        """
        execute_db("""
            INSERT INTO demand_predictions (
                food_item_id, prediction_date, model_used, predicted_demand,
                recommended_prep_qty, safety_stock_used, usable_inventory_used,
                risk_level, estimated_waste_risk
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            food_item_id, target_date, model_used, predicted_demand,
            recommended_prep, safety_stock, usable_inventory, risk_level, estimated_waste
        ))
