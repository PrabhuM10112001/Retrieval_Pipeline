from pydantic import BaseModel
from typing import Optional

class FilterRequest(BaseModel):
    source: Optional[str] = None
    topic: Optional[str] = None
    msgtimestamp: Optional[int] = None
    msgdate: Optional[str] = None
    orgid: Optional[int] = None
    resellerid: Optional[int] = None
    customerid: Optional[int] = None
    dealerid: Optional[int] = None
    regionid: Optional[int] = None
    createddate: Optional[int] = None


class AskRequest(BaseModel):
    query: str
    k: int = 20
    collectionname: str
    filter: Optional[FilterRequest] = None
    session_id: Optional[str] = "default"


class AskVehicleRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class VehicleSummaryByNameRequest(BaseModel):
    query: str
    vehicleid_collection: str 
    vehicle_name_key: str = "vehicleNo"
    vehicle_id_key: str = "vehicleid"
    k: int = 5
    session_id: Optional[str] = "default"


class UploadTxtRequest(BaseModel):
    file_path: str  
class UploadCsvRequest(BaseModel):
    file_path: str
    text_column: Optional[str] = None
