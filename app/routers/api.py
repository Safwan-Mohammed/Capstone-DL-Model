from fastapi import APIRouter, Body
import asyncio
from typing import List, Dict
from app.services.transformer_model_service import get_model_prediction
from app.psetae_model.npy_conversion import convert_to_npy

router = APIRouter()

@router.post("/crop-prediction-transformer")
async def crop_prediction_endpoint(
    data: List[Dict] = Body(..., description="List of records representing a DataFrame")
):
    results = await asyncio.to_thread(get_model_prediction, data)
    return results

@router.post("/crop-prediction-psetae")
async def crop_prediction_endpoint(
    data: List[Dict] = Body(..., description="List of records representing a DataFrame")
):
    try:

        await asyncio.to_thread(convert_to_npy, data, "../PRED_DATA")

        predictions = await asyncio.to_thread(get_model_prediction, data)

        return {"predictions": predictions}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}