from fastapi import APIRouter, HTTPException, Depends
from rag_engine import RAGEngine
from Schemas import AskVehicleRequest, AskRequest, VehicleSummaryByNameRequest
from JwtValidation.JwtTokenValidate import jwt_filter
from vehicle_summary_all_batched import VehicleSummaryAllBatchProcessor

import re

router = APIRouter(
    prefix="/rag",
    tags=["RAG APIs"],
    dependencies=[Depends(jwt_filter)]  
)

RAG = RAGEngine()
vehiclesummaryAll = VehicleSummaryAllBatchProcessor()
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


@router.get("/get_userrole")
def getuserrole(claims: dict = Depends(jwt_filter)):

    response = RAG.get_userrole(claims=claims)
    return response



         

@router.post("/ask_RAG")
def ask(req: AskRequest):
    collectionName = req.collectionname  
    RAG1 = RAGEngine(collection_name=collectionName)
    out = RAG1.answer_question(req)
    return out

@router.post("/ask_Vehicles")
def add_json(req: AskVehicleRequest):
    result = RAG.answer_vehicle(req.query, session_id=req.session_id)
    return result


@router.post("/chatBot_RAG")
def chatbot_rag_method(
    req: VehicleSummaryByNameRequest,
    claims: dict = Depends(jwt_filter),
):
    session_id = req.session_id or "default"
    result = RAG.query_status_check(
        query=req.query,
        session_id=session_id,
        use_history_for_classification=True,
    )

    if result == "generic":
        vehicle_result = RAG.answer_vehicle(
            query=req.query,
            session_id="generic",
            persist_history=False,
        )
        response_payload = {
            "query": req.query,
            "session_id": session_id,
            "response": vehicle_result,
            "results": [],
        }
    elif result == "vehicledetail" :

        response_payload = RAG.get_vehicle_detail(
            query=req.query,
            vehicle_detail_collection=req.vehicleid_collection,
            session_id="vehicledetail",
            claims=claims,
            persist_history=False,
        )
         

    elif result in {"vehiclesummary-notall"}:
        response_payload = RAG.get_vehicle_summary_by_name(
            query=req.query,
            vehicleid_collection=req.vehicleid_collection,
            vehicle_name_key="VEHICLE_NO",
            vehicle_id_key="VEHICLE_ID",
            k=req.k,
            session_id="vehiclesummary-all",
            claims=claims,
            persist_history=False,
        )

    elif result in {"vehiclesummary-all"}:

        batch_result = vehiclesummaryAll.process_all_vehicle_summary(
            query=req.query,
            vehicleid_collection=req.vehicleid_collection,
            # vehicle_name_key="VEHICLE_NO",
            # vehicle_id_key="VEHICLE_ID",
            # k=req.k,
            session_id=session_id,
            claims=claims,
        )
        response_payload = {
            "query": req.query,
            "session_id": session_id,
            "response": batch_result["final_summary"],
            "results": batch_result["final_summary"],
        }


    else:
        response_payload = {
            "query": req.query,
            "response": "Sorry, I couldn't assist you for this query. Please ask a different question.",
            "results": [],
        }

    assistant_history_text = RAG._response_to_history_text(response_payload)
    RAG.append_history_turn(
        session_id=session_id,
        user_text=req.query,
        assistant_text=assistant_history_text,
    )
    return response_payload


