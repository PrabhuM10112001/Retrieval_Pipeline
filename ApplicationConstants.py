import os


class AppConstants:

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    VECTOR_DB_LOC = "/home/prabhu/Music/Vector_DB"
    # OBD_LOGIN_URL = "https://ukapi.nesh.live:9443/authenticate"
    # OBD_LOGIN_CREDENTIAL = {
    # "userName": "devopsuk@gmail.com",
    # "password": "Nesh@1234"}
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o-mini"
    GEMINI_MODEL = "gemini-2.0-flash"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SECRET_KEY = os.getenv("JWT_SECRET_KEY", "neshauthkey")

    
