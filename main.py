from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from logger import logger 
from rag_engine import RAGEngine
from fastapi.middleware.cors import CORSMiddleware
from Schemas import AskVehicleRequest , FilterRequest, AskRequest
from fastapi import APIRouter
from rag_router import router as rag_router


app = FastAPI(title="RAG System Retrieval API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rag_router)



