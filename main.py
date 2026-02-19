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


app = FastAPI(title="RAG FastAPI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(rag_router)

# @rag_router.get("/")
# def root():
#     return {"message": "RAG FastAPI is running!"}

# @rag_router.post("/upload_txt")


# # @app.post("/upload_csv")
# # def upload_csv(req: UploadCsvRequest):
# #     if not os.path.isfile(req.file_path):
# #         raise HTTPException(status_code=400, detail="file_path does not exist")
    
# #     RAG1 = RAGEngine(
# #     collection_name="csv_data",
# #     persist_directory="data/vector_store",
# #     embedding_model="all-MiniLM-L6-v2",    
# # )
# #     res = RAG1.add_csv(req.file_path,text_column=req.text_column or "")
# #     return {"status": "ok", "result": res}

# @rag_router.post("/ask_RAG")
# def ask(req: AskRequest):
#     collectionName = req.collectionname  
#     RAG1 = RAGEngine(collection_name=collectionName)
#     out = RAG1.answer_question(req)
#     return out


# @rag_router.post("/ask_Vehicles")
# def add_json(req: AskVehicleRequest):
#     result = RAG.answer_vehicle(req)
#     return result



