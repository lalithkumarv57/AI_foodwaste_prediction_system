import os
import sys
import unittest
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config
from database.db import get_db_connection, query_db, init_db
from ml.dataset_generator import generate_historical_dataset
from ml.preprocessor import FoodDemandPreprocessor
from ml.model_trainer import train_and_evaluate_models
from ml.predictor import DemandPredictor
from app import app

class TestFoodWastePredictionSystem(unittest.TestCase):
    """
    Unit and integration test suite for AI-based Food Waste Prediction System.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test environment and initialize Flask test client."""
        cls.client = app.test_client()
        cls.predictor = DemandPredictor()

    def test_01_database_connectivity(self):
        """Verify database connection and required tables exist."""
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row['name'] for row in cur.fetchall()]
        conn.close()

        required_tables = [
            'food_items', 'inventory', 'food_preparation',
            'daily_sales', 'food_waste_records', 'demand_predictions',
            'model_evaluation_results'
        ]
        for t in required_tables:
            self.assertIn(t, tables, f"Missing table {t} in database")

    def test_02_food_items_loaded(self):
        """Verify food items and inventory records are populated."""
        items = query_db("SELECT * FROM food_items")
        self.assertGreater(len(items), 0, "No food items found in database")
        
        inventory = query_db("SELECT * FROM inventory")
        self.assertGreater(len(inventory), 0, "No inventory records found")

    def test_03_csv_validation_logic(self):
        """Test dataset validation and missing column checks."""
        import pandas as pd
        valid_df = pd.DataFrame({
            'Date': ['2026-10-01', '2026-10-02'],
            'Food_Item': ['Vegetable Biryani', 'Vegetable Biryani'],
            'Category': ['Rice & Grains', 'Rice & Grains'],
            'Quantity_Sold': [80, 85]
        })
        is_valid, issues, clean_df = FoodDemandPreprocessor.validate_csv(valid_df)
        self.assertTrue(is_valid, f"Validation failed on valid dataframe: {issues}")

        # Invalid dataframe missing Quantity_Sold
        invalid_df = pd.DataFrame({
            'Date': ['2026-10-01'],
            'Food_Item': ['Pasta Primavera']
        })
        is_invalid, issues, _ = FoodDemandPreprocessor.validate_csv(invalid_df)
        self.assertFalse(is_invalid, "Validation should fail for missing required column")

    def test_04_feature_engineering_no_leakage(self):
        """Verify time features and lag shifts do not leak future information."""
        import pandas as pd
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2026-09-01', periods=10),
            'Food_Item': ['Biryani'] * 10,
            'Category': ['Rice'] * 10,
            'Quantity_Sold': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        })
        prep = FoodDemandPreprocessor()
        engineered = prep.engineer_features(sample_df, is_training=True)

        # On index 1, lag 1 must be the previous day's demand (50)
        self.assertEqual(engineered.loc[1, 'Lag_1'], 50)
        # On index 2, lag 1 must be 55
        self.assertEqual(engineered.loc[2, 'Lag_1'], 55)

    def test_05_recommendation_formula(self):
        """
        Verify formula: Recommended Preparation Quantity = MAX(0, Predicted Demand + Safety Stock - Usable Inventory)
        """
        pred_demand = 100.0
        usable_inv = 20.0
        safety_buffer_pct = 10.0 # 10% safety stock = 10.0
        
        rec = DemandPredictor.calculate_recommendation(
            predicted_demand=pred_demand,
            usable_inventory=usable_inv,
            safety_buffer_pct=safety_buffer_pct
        )
        # Safety stock = 100 * 0.10 = 10
        # Recommended prep = MAX(0, 100 + 10 - 20) = 90
        self.assertEqual(rec['safety_stock'], 10.0)
        self.assertEqual(rec['recommended_prep'], 90.0)

        # Test case where inventory exceeds demand + safety stock
        rec_excess = DemandPredictor.calculate_recommendation(
            predicted_demand=50.0,
            usable_inventory=100.0,
            safety_buffer_pct=10.0
        )
        self.assertEqual(rec_excess['recommended_prep'], 0.0)

    def test_06_waste_risk_analysis(self):
        """Verify waste risk calculation and high/medium/low risk categorization."""
        # High overproduction test
        risk_high = DemandPredictor.analyze_waste_risk(
            planned_preparation=150.0,
            predicted_demand=100.0,
            usable_inventory=0.0,
            unit_cost=50.0
        )
        self.assertEqual(risk_high['risk_level'], 'High')
        self.assertEqual(risk_high['estimated_waste_portions'], 50.0)
        self.assertEqual(risk_high['estimated_waste_cost'], 2500.0)

        # Lean low waste test
        risk_low = DemandPredictor.analyze_waste_risk(
            planned_preparation=102.0,
            predicted_demand=100.0,
            usable_inventory=0.0,
            unit_cost=50.0
        )
        self.assertEqual(risk_low['risk_level'], 'Low')

    def test_07_api_predict_endpoint(self):
        """Verify POST /api/predict returns valid JSON response."""
        payload = {
            'food_item_id': 1,
            'target_date': '2026-10-20',
            'weather': 'Sunny',
            'temperature': 25.0,
            'is_holiday': 0,
            'special_event': 0,
            'usable_inventory': 10,
            'safety_buffer_pct': 10,
            'model_choice': 'best'
        }
        res = cls_client = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('recommendation', data)
        self.assertIn('waste_risk', data)
        self.assertGreaterEqual(data['prediction']['predicted_demand'], 0)

    def test_08_web_routes_status_codes(self):
        """Verify all core view routes respond with HTTP 200 OK."""
        routes = ['/', '/predict', '/analysis', '/inventory', '/models', '/upload', '/reports']
        for route in routes:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed with status {res.status_code}")

if __name__ == '__main__':
    unittest.main()
