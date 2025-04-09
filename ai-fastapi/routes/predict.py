from fastapi import APIRouter
from models import SalesRequest, MarketRequest
from utils.tf_model import predict_sales, predict_market_sales

router = APIRouter()

@router.post("/predict")
async def predict_sales_route(request: SalesRequest):
    try:
        result = predict_sales(request.district, request.menu, request.date)
        return {"predicted_sales": int(result)}
    except Exception as e:
        return {"error": str(e)}

@router.get("/market-predict")
async def predict_market_route(district: str, category: str, year: int, month: int):
    try:
        result = predict_market_sales(district, category, year, month)
        return {"predicted_sales": int(result)}
    except Exception as e:
        return {"error": str(e)}
