from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import json
import os
from datetime import datetime
import tensorflow as tf

app = FastAPI()

# 경로 설정
SALES_MODEL_PATH = "saved_model/sales_model.h5"
CATEGORY_PATH = "saved_model/feature_categories.json"
MARKET_MODEL_PATH = "saved_market_model/market_model.h5"
MARKET_FEATURE_PATH = "saved_market_model/market_features.json"


# 🔹 입력 스키마
class SalesRequest(BaseModel):
    district: str
    menu: str
    date: str


class MarketRequest(BaseModel):
    district: str
    category: str
    year: int
    month: int


# 🔹 예측 함수
def predict_sales(district: str, menu: str, date_str: str) -> float:
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
    prediction = model.predict(input_df)
    return float(prediction[0][0])


def predict_market_sales(district: str, category: str, year: int, month: int) -> float:
    with open(MARKET_FEATURE_PATH, "r") as f:
        features = json.load(f)

    input_data = {"year": year, "month": month}
    for d in features["district"]:
        input_data[f"district_{d}"] = 1 if d == district else 0
    for c in features["category"]:
        input_data[f"category_{c}"] = 1 if c == category else 0

    df_input = pd.DataFrame([input_data])
    model = tf.keras.models.load_model(MARKET_MODEL_PATH)
    pred = model.predict(df_input)
    return float(pred[0][0])


# 🔹 라우팅
@app.post("/predict")
async def predict_sales_api(request: SalesRequest):
    try:
        result = predict_sales(request.district, request.menu, request.date)
        return {"predicted_sales": int(result)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/market-predict")
async def market_forecast_api(district: str, category: str, year: int, month: int):
    try:
        result = predict_market_sales(district, category, year, month)
        return {"predicted_sales": int(result)}
    except Exception as e:
        return {"error": str(e)}
