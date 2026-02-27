from fastapi import APIRouter, HTTPException, Depends
from rag_engine import RAGEngine
from Schemas import AskVehicleRequest, AskRequest, VehicleSummaryByNameRequest
from JwtValidation.JwtTokenValidate import jwt_filter
import re

router = APIRouter(
    prefix="/rag",
    tags=["RAG APIs"],
    dependencies=[Depends(jwt_filter)]  
)

RAG = RAGEngine()

# def is_vehicle_related_query(query: str) -> bool:
#     if not query or not query.strip():
#         return False

#     normalized = query.lower()
#     vehicle_no_pattern = r"[A-Za-z]{2}\d{1,2}[A-Za-z]{1,3}\d{1,4}"
#     if re.search(vehicle_no_pattern, query.replace(" ", "")):
#         return True

#     keywords = [
#         "vehicle",
#         "car",
#         "truck",
#         "bus",
#         "bike",
#         "fleet",
#         "mileage",
#         "distance",
#         "running cost",
#         "fuel",
#         "engine",
#         "gps",
#         "telematics",
#         "trip",
#         "idle",
#         "stoppage",
#         "odometer",
#     ]
#     return any(keyword in normalized for keyword in keywords)






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


@router.post("/chatBot_RAG")
def chatbot_rag_method(
    req: VehicleSummaryByNameRequest,
    claims: dict = Depends(jwt_filter),
):
        
        result = RAG.query_status_check(req.query)
        

        if result == "Generic" or result == "generic":

             vehicleResult = RAG.answer_vehicle(req.query) 

             return {

            "query": req.query,
            "session_id": req.session_id,
            "response": vehicleResult,
            "results": [],
        }


        elif result == "Vehiclesummary-all" or result == "vehiclesummary-all" or result == "Vehiclesummary-notall" or result == "vehiclesummary-notall":

        #      return {
        #     "query": req.query,
        #     "session_id": req.session_id,
        #     "response": "Please ask only generic vehicle-related questions.",
        #     "results": [],
        # }
        # return {
        #     "query": req.query,
        #     "session_id": req.session_id,
        #     "response": "Please ask only generic vehicle-related questions.",
        #     "results": [],
        # }

            vehicleSummaryResult = RAG.get_vehicle_summary_by_name(
        query=req.query,
        vehicleid_collection=req.vehicleid_collection,
        summary_collection=req.summary_collection,
        vehicle_name_key="VEHICLE_NO",
        vehicle_id_key="VEHICLE_ID",
        k=req.k,
        session_id=req.session_id,
        claims=claims,
    )
            return vehicleSummaryResult
        
        elif result == "none" or result == "None":

             return {
            "query": req.query,
            "session_id": req.session_id,
            "response": "Sorry, I couldn't assist you for this query. Please ask a different question.",
            "results": [],
        }
