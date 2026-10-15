-- AI-Based Food Waste Prediction System
-- SQLite Database Schema Definition
-- Supporting SDG 12: Responsible Consumption and Production

PRAGMA foreign_keys = ON;

-- 1. Food Items Catalog Table
CREATE TABLE IF NOT EXISTS food_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    unit TEXT NOT NULL DEFAULT 'portions',
    unit_cost REAL NOT NULL CHECK(unit_cost >= 0),
    selling_price REAL NOT NULL CHECK(selling_price >= 0),
    shelf_life_days INTEGER NOT NULL DEFAULT 1 CHECK(shelf_life_days >= 1),
    default_safety_buffer_pct REAL NOT NULL DEFAULT 10.0 CHECK(default_safety_buffer_pct >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Current Inventory Table
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_item_id INTEGER NOT NULL UNIQUE,
    current_usable_stock REAL NOT NULL DEFAULT 0 CHECK(current_usable_stock >= 0),
    safety_stock_buffer REAL NOT NULL DEFAULT 0 CHECK(safety_stock_buffer >= 0),
    storage_location TEXT DEFAULT 'Main Kitchen Fridge/Pantry',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE CASCADE
);

-- 3. Food Preparation Logs
CREATE TABLE IF NOT EXISTS food_preparation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_item_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    meal_session TEXT DEFAULT 'All-Day',
    quantity_prepared REAL NOT NULL CHECK(quantity_prepared >= 0),
    preparation_cost REAL NOT NULL CHECK(preparation_cost >= 0),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE CASCADE
);

-- 4. Daily Food Sales Logs
CREATE TABLE IF NOT EXISTS daily_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_item_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    meal_session TEXT DEFAULT 'All-Day',
    quantity_sold REAL NOT NULL CHECK(quantity_sold >= 0),
    revenue REAL NOT NULL CHECK(revenue >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE CASCADE
);

-- 5. Food Waste Records Table
CREATE TABLE IF NOT EXISTS food_waste_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_item_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    meal_session TEXT DEFAULT 'All-Day',
    quantity_wasted REAL NOT NULL CHECK(quantity_wasted >= 0),
    waste_cost REAL NOT NULL CHECK(waste_cost >= 0),
    waste_reason TEXT DEFAULT 'Overproduction / Unsold Food',
    logged_by TEXT DEFAULT 'Operations Staff',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE CASCADE
);

-- 6. Machine Learning Demand Predictions History
CREATE TABLE IF NOT EXISTS demand_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    food_item_id INTEGER NOT NULL,
    prediction_date TEXT NOT NULL,
    model_used TEXT NOT NULL,
    predicted_demand REAL NOT NULL CHECK(predicted_demand >= 0),
    recommended_prep_qty REAL NOT NULL CHECK(recommended_prep_qty >= 0),
    safety_stock_used REAL NOT NULL DEFAULT 0,
    usable_inventory_used REAL NOT NULL DEFAULT 0,
    risk_level TEXT NOT NULL, -- Low, Medium, High
    estimated_waste_risk REAL NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (food_item_id) REFERENCES food_items (id) ON DELETE CASCADE
);

-- 7. Model Evaluation Results
CREATE TABLE IF NOT EXISTS model_evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    mape REAL NOT NULL,
    r2_score REAL NOT NULL,
    sample_size INTEGER NOT NULL,
    best_model_flag INTEGER DEFAULT 0,
    feature_importance_json TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_food_prep_date ON food_preparation (date);
CREATE INDEX IF NOT EXISTS idx_food_sales_date ON daily_sales (date);
CREATE INDEX IF NOT EXISTS idx_food_waste_date ON food_waste_records (date);
CREATE INDEX IF NOT EXISTS idx_demand_pred_date ON demand_predictions (prediction_date);
