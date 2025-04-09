import json
import os
import pandas as pd
import tensorflow as tf
from datetime import datetime

# 경로 상수
SALES_MODEL_PATH = "saved_model/sales_model.h5"
CATEGORY_PATH = "saved_model/feature_categories.json"
MARKET_MODEL_PATH = "saved_market_model/market_model.h5"
MARKET_FEATURE_PATH = "saved_market_model/market_features.json"

def predict_sales(district, menu, date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    month = date_obj.month
    weekday = date_obj.strftime("%A")

    with open(CATEGORY_PATH, "r") as f:
        categories = json.load(f)

    input_dict = {"month": month}
    for d in categories["district"]:
        input_dict[f"district_{d}"] = 1 if d == district else 0
    for m in categories["menu"]:
        input_dict[f"menu_{m}"] = 1 if m == menu else 0
    for w in categories["weekday"]:
        input_dict[f"weekday_{w}"] = 1 if w == weekday else 0

    input_df = pd.DataFrame([input_dict])
    model = tf.keras.models.load_model(SALES_MODEL_PATH)
    return float(model.predict(input_df)[0][0])

def predict_market_sales(district, category, year, month):
    with open(MARKET_FEATURE_PATH, "r") as f:
        features = json.load(f)

    input_data = {"year": year, "month": month}
    for d in features["district"]:
        input_data[f"district_{d}"] = 1 if d == district else 0
    for c in features["category"]:
        input_data[f"category_{c}"] = 1 if c == category else 0

    input_df = pd.DataFrame([input_data])
    model = tf.keras.models.load_model(MARKET_MODEL_PATH)
    return float(model.predict(input_df)[0][0])
