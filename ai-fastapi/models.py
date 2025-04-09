from pydantic import BaseModel

class SalesRequest(BaseModel):
    district: str
    menu: str
    date: str

class MarketRequest(BaseModel):
    district: str
    category: str
    year: int
    month: int
