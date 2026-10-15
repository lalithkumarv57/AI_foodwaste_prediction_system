import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import random
import datetime
import numpy as np
import pandas as pd
from config import Config

# Standard catalog of food items across diverse categories
DEFAULT_FOOD_ITEMS = [
    {
        "name": "Vegetable Biryani",
        "category": "Rice & Grains",
        "unit": "portions",
        "unit_cost": 45.0,
        "selling_price": 120.0,
        "base_demand": 85,
        "shelf_life_days": 1,
        "weekend_boost": 1.25,
        "rainy_boost": 1.15
    },
    {
        "name": "Paneer Butter Masala",
        "category": "Curry & Stew",
        "unit": "portions",
        "unit_cost": 55.0,
        "selling_price": 150.0,
        "base_demand": 70,
        "shelf_life_days": 1,
        "weekend_boost": 1.30,
        "rainy_boost": 1.10
    },
    {
        "name": "Grilled Chicken Salad",
        "category": "Salads & Bowls",
        "unit": "bowls",
        "unit_cost": 60.0,
        "selling_price": 180.0,
        "base_demand": 50,
        "shelf_life_days": 1,
        "weekend_boost": 0.90,  # Higher on weekdays (office lunch)
        "rainy_boost": 0.80,   # Lower on cold/rainy days
        "hot_boost": 1.25
    },
    {
        "name": "Pasta Primavera",
        "category": "Main Course",
        "unit": "plates",
        "unit_cost": 50.0,
        "selling_price": 140.0,
        "base_demand": 60,
        "shelf_life_days": 1,
        "weekend_boost": 1.35,
        "rainy_boost": 1.05
    },
    {
        "name": "Lentil Soup (Dal Tadka)",
        "category": "Soups & Sides",
        "unit": "bowls",
        "unit_cost": 25.0,
        "selling_price": 70.0,
        "base_demand": 95,
        "shelf_life_days": 1,
        "weekend_boost": 1.05,
        "rainy_boost": 1.30
    },
    {
        "name": "Margherita Flatbread Pizza",
        "category": "Bakery & Pizza",
        "unit": "units",
        "unit_cost": 40.0,
        "selling_price": 130.0,
        "base_demand": 65,
        "shelf_life_days": 1,
        "weekend_boost": 1.45,
        "rainy_boost": 1.10
    },
    {
        "name": "Steamed Basmati Rice",
        "category": "Rice & Grains",
        "unit": "portions",
        "unit_cost": 15.0,
        "selling_price": 50.0,
        "base_demand": 120,
        "shelf_life_days": 1,
        "weekend_boost": 1.10,
        "rainy_boost": 1.05
    },
    {
        "name": "Mango Yogurt Parfait",
        "category": "Desserts & Dairy",
        "unit": "cups",
        "unit_cost": 30.0,
        "selling_price": 90.0,
        "base_demand": 40,
        "shelf_life_days": 2,
        "weekend_boost": 1.30,
        "rainy_boost": 0.75,
        "hot_boost": 1.35
    }
]

WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Cold", "Hot"]
WEATHER_WEIGHTS = [0.45, 0.25, 0.15, 0.08, 0.07]

def generate_historical_dataset(num_days=180, end_date=None, output_csv=None):
    """
    Generates a realistic historical dataset for food demand and waste prediction.
    Incorporates:
    - Day-of-week demand patterns (weekday vs weekend)
    - National and local holidays
    - Special events (festivals, campus events, sports screenings)
    - Weather patterns and temperature
    - Typical kitchen overproduction behavior leading to real-world food waste
    """
    if end_date is None:
        end_date = datetime.date.today()
    elif isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    start_date = end_date - datetime.timedelta(days=num_days - 1)
    
    # Pre-select some random holidays and special event dates
    all_dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    holiday_dates = set(random.sample(all_dates, max(3, num_days // 25)))
    event_dates = set(random.sample(all_dates, max(5, num_days // 15)))

    records = []
    
    # Seed for reproducibility with slight variation
    np.random.seed(42)
    random.seed(42)

    for current_date in all_dates:
        date_str = current_date.strftime("%Y-%m-%d")
        day_of_week = current_date.strftime("%A")
        is_weekend = 1 if current_date.weekday() >= 5 else 0
        is_holiday = 1 if current_date in holiday_dates else 0
        special_event = 1 if current_date in event_dates else 0
        
        # Season / Weather simulation
        month = current_date.month
        if month in [12, 1, 2]:
            weather = random.choices(["Cold", "Sunny", "Cloudy"], weights=[0.5, 0.3, 0.2])[0]
            temp_c = round(random.uniform(14.0, 22.0), 1)
        elif month in [3, 4, 5]:
            weather = random.choices(["Sunny", "Hot", "Cloudy"], weights=[0.5, 0.35, 0.15])[0]
            temp_c = round(random.uniform(28.0, 39.0), 1)
        elif month in [6, 7, 8, 9]:
            weather = random.choices(["Rainy", "Cloudy", "Sunny"], weights=[0.45, 0.35, 0.2])[0]
            temp_c = round(random.uniform(24.0, 31.0), 1)
        else: # Oct, Nov
            weather = random.choices(["Sunny", "Cloudy", "Rainy"], weights=[0.6, 0.25, 0.15])[0]
            temp_c = round(random.uniform(20.0, 29.0), 1)

        for item in DEFAULT_FOOD_ITEMS:
            base = item["base_demand"]
            multiplier = 1.0
            
            # Weekend effect
            if is_weekend:
                multiplier *= item.get("weekend_boost", 1.0)
            
            # Holiday effect (families eat out, office crowds drop)
            if is_holiday:
                if item["category"] in ["Main Course", "Rice & Grains", "Bakery & Pizza"]:
                    multiplier *= 1.20
                else:
                    multiplier *= 0.90
                    
            # Special event effect (major traffic surge)
            if special_event:
                multiplier *= 1.30

            # Weather effects
            if weather == "Rainy":
                multiplier *= item.get("rainy_boost", 1.0)
            elif weather == "Hot":
                multiplier *= item.get("hot_boost", 1.0)
            elif weather == "Cold" and item["category"] in ["Soups & Sides", "Curry & Stew"]:
                multiplier *= 1.20

            # Random variance (simulates genuine stochastic consumer behavior)
            noise = np.random.normal(loc=0.0, scale=0.08)
            expected_demand = base * multiplier * (1.0 + noise)
            quantity_sold = max(5, int(round(expected_demand)))

            # Kitchen preparation behavior without AI:
            # Kitchens traditionally overproduce by 10-25% to avoid stockouts,
            # or misestimate on weekends/events, leading to avoidable waste!
            traditional_prep_buffer = np.random.uniform(1.08, 1.28)
            if special_event or is_weekend:
                # Often over-prepared on uncertain busy days
                traditional_prep_buffer += np.random.uniform(0.05, 0.15)
                
            quantity_prepared = int(round(quantity_sold * traditional_prep_buffer))
            # Occasional small shortage (1-2% probability)
            if random.random() < 0.03:
                quantity_prepared = max(5, int(round(quantity_sold * 0.92)))
                quantity_sold = quantity_prepared  # sold out

            quantity_wasted = max(0, quantity_prepared - quantity_sold)
            unit_cost = item["unit_cost"]
            selling_price = item["selling_price"]

            records.append({
                "Date": date_str,
                "Food_Item": item["name"],
                "Category": item["category"],
                "Day_Of_Week": day_of_week,
                "Is_Weekend": is_weekend,
                "Is_Holiday": is_holiday,
                "Special_Event": special_event,
                "Weather_Condition": weather,
                "Temperature_C": temp_c,
                "Quantity_Prepared": quantity_prepared,
                "Quantity_Sold": quantity_sold,
                "Quantity_Wasted": quantity_wasted,
                "Unit_Cost": unit_cost,
                "Selling_Price": selling_price
            })

    df = pd.DataFrame(records)
    
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Generated synthetic historical dataset with {len(df)} records at: {output_csv}")
        
    return df

if __name__ == '__main__':
    csv_path = os.path.join(Config.DATA_DIR, 'historical_food_waste_data.csv')
    generate_historical_dataset(num_days=180, output_csv=csv_path)
