import os


class AppConstants:

    OPEN_API_KEY = os.getenv("OPENAI_API_KEY", "")
    VECTOR_DB_LOC = "/home/prabhu/Music/Vector_DB"
    # OBD_LOGIN_URL = "https://ukapi.nesh.live:9443/authenticate"
    # OBD_LOGIN_CREDENTIAL = {
    # "userName": "devopsuk@gmail.com",
    # "password": "Nesh@1234"}
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SECRET_KEY = "neshauthkey"

    
