import sqlite3
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from config import Config

def get_db_connection():
    """Returns a SQLite connection with row factory configured to return dict-like rows."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db(force_recreate=False):
    """Initializes the database schema."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    
    if force_recreate and os.path.exists(Config.DATABASE_PATH):
        try:
            os.remove(Config.DATABASE_PATH)
        except Exception as e:
            print(f"Warning removing existing database: {e}")

    conn = get_db_connection()
    with open(Config.SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def query_db(query, args=(), one=False):
    """Helper to query the database and return dictionaries."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, args)
    rv = cur.fetchall()
    conn.close()
    if rv:
        if one:
            return dict(rv[0])
        return [dict(row) for row in rv]
    return None if one else []

def execute_db(query, args=(), commit=True):
    """Helper to execute INSERT, UPDATE, DELETE queries and return lastrowid."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, args)
    if commit:
        conn.commit()
    last_id = cur.lastrowid
    affected = cur.rowcount
    conn.close()
    return last_id or affected

def execute_many_db(query, param_list):
    """Helper to execute batch queries with executemany."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(query, param_list)
    conn.commit()
    rowcount = cur.rowcount
    conn.close()
    return rowcount

def seed_database_from_csv(csv_path=None):
    """Loads historical food sales, preparation, and waste data into SQLite."""
    if csv_path is None:
        csv_path = os.path.join(Config.DATA_DIR, 'historical_food_waste_data.csv')
        
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Please generate it first.")
        return False
        
    df = pd.read_csv(csv_path)
    init_db(force_recreate=True)
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Insert unique Food Items
    unique_items = df[['Food_Item', 'Category', 'Unit_Cost', 'Selling_Price']].drop_duplicates(subset=['Food_Item'])
    item_id_map = {}
    
    for _, row in unique_items.iterrows():
        cur.execute("""
            INSERT INTO food_items (name, category, unit_cost, selling_price, shelf_life_days, default_safety_buffer_pct)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row['Food_Item'], row['Category'], float(row['Unit_Cost']), float(row['Selling_Price']), 1, 12.0))
        item_id_map[row['Food_Item']] = cur.lastrowid
        
        # 2. Initialize inventory for each item
        cur.execute("""
            INSERT INTO inventory (food_item_id, current_usable_stock, safety_stock_buffer, storage_location)
            VALUES (?, ?, ?, ?)
        """, (cur.lastrowid, 15.0, 10.0, 'Kitchen Main Storage'))
        
    # 3. Batch insert food preparation, sales, and waste
    prep_records = []
    sales_records = []
    waste_records = []
    
    for _, row in df.iterrows():
        item_id = item_id_map[row['Food_Item']]
        date = str(row['Date'])
        prep_qty = float(row['Quantity_Prepared'])
        sold_qty = float(row['Quantity_Sold'])
        wasted_qty = float(row['Quantity_Wasted'])
        unit_cost = float(row['Unit_Cost'])
        selling_price = float(row['Selling_Price'])
        
        prep_records.append((item_id, date, 'All-Day', prep_qty, prep_qty * unit_cost, 'Daily planned prep'))
        sales_records.append((item_id, date, 'All-Day', sold_qty, sold_qty * selling_price))
        if wasted_qty > 0:
            waste_records.append((item_id, date, 'All-Day', wasted_qty, wasted_qty * unit_cost, 'Overproduction / Unsold Food', 'System Sync'))
            
    cur.executemany("""
        INSERT INTO food_preparation (food_item_id, date, meal_session, quantity_prepared, preparation_cost, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, prep_records)
    
    cur.executemany("""
        INSERT INTO daily_sales (food_item_id, date, meal_session, quantity_sold, revenue)
        VALUES (?, ?, ?, ?, ?)
    """, sales_records)
    
    cur.executemany("""
        INSERT INTO food_waste_records (food_item_id, date, meal_session, quantity_wasted, waste_cost, waste_reason, logged_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, waste_records)
    
    conn.commit()
    conn.close()
    print(f"Database seeded successfully with {len(unique_items)} items and {len(df)} daily activity records.")
    return True

if __name__ == '__main__':
    seed_database_from_csv()

