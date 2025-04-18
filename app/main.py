from fastapi import FastAPI
from app.routers import api
import os
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

app = FastAPI(title="DL Model Microservice")
app.include_router(api.router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=6000, reload=True, workers=2)

""" 
RUN THIS FILE USING THE COMMAND 
    python -m app.main 
"""