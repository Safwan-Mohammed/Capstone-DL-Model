from fastapi import APIRouter, Body
import asyncio
from typing import List, Dict
from app.services.model_service import get_model_prediction

router = APIRouter()

@router.post("/crop-prediction")
async def crop_prediction_endpoint(
    data: List[Dict] = Body(..., description="List of records representing a DataFrame")
):
    results = await asyncio.to_thread(get_model_prediction, data)
    return results