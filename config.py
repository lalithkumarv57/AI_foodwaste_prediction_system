import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'food-waste-prediction-secret-key-2026'
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'food_waste.db')
    SCHEMA_PATH = os.path.join(BASE_DIR, 'database', 'schema.sql')
    MODEL_DIR = os.path.join(BASE_DIR, 'ml', 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {'csv'}
