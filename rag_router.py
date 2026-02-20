from fastapi import APIRouter, HTTPException, Depends
from rag_engine import RAGEngine
from Schemas import AskVehicleRequest, AskRequest, VehicleSummaryByNameRequest
from JwtValidation.JwtTokenValidate import jwt_filter

router = APIRouter(
    prefix="/rag",
    tags=["RAG APIs"],
    dependencies=[Depends(jwt_filter)]  
)

RAG = RAGEngine()

@router.get("/")
def root():
    return {"message": "RAG FastAPI is running!"} 
     



@router.post("/ask_RAG")
def ask(req: AskRequest):
    collectionName = req.collectionname  
    RAG1 = RAGEngine(collection_name=collectionName)
    out = RAG1.answer_question(req)
    return out

@router.post("/ask_Vehicles")
def add_json(req: AskVehicleRequest):
    result = RAG.answer_vehicle(req)
    return result


@router.post("/vehiclesummary")
def vehicle_summary_by_name(
    req: VehicleSummaryByNameRequest,
    claims: dict = Depends(jwt_filter),
):
    result = RAG.get_vehicle_summary_by_name(
        query=req.query,
        vehicleid_collection=req.vehicleid_collection,
        summary_collection=req.summary_collection,
        vehicle_name_key=req.vehicle_name_key,
        vehicle_id_key=req.vehicle_id_key,
        k=req.k,
        session_id=req.session_id,
        claims=claims,
    )
    return result
