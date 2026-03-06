# rag_engine.py
import os
import re
from datetime import datetime

from typing import List, Dict, Any, Optional, Sequence
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
try:
    import tiktoken
except ImportError:
    tiktoken = None
from langchain_core.prompts import PromptTemplate
from ApplicationConstants import AppConstants
from Schemas import AskRequest, AskVehicleRequest
from logger import logger
class RAGEngine:
    _chat_histories:  Dict[str, InMemoryChatMessageHistory] = {}
    _CONTEXT_BUDGET_TOKENS = 12000
    _RESERVED_COMPLETION_TOKENS = 500
    _MAX_RESPONSE_TOKENS = 2000

    def __init__(
        self,
        collection_name: str = "t_defaultcollection",
        persist_directory: str = AppConstants.VECTOR_DB_LOC,
        embedding_model: str = AppConstants.EMBEDDING_MODEL,
        # llm_model: Optional[str] = None,
        # llm_provider: str = AppConstants.LLM_PROVIDER,
      
    ):
        
        # self.system_message = SystemMessage(content=rules)   
        # self.conversation_history = []
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)

        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedding_function
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        # selected_provider = (llm_provider or "openai").strip().lower()
        # if selected_provider == "gemini":
        #     selected_model = llm_model or AppConstants.GEMINI_MODEL
        #     if ChatGoogleGenerativeAI is None:
        #         raise ImportError(
        #             "Gemini provider selected, but langchain_google_genai is not installed."
        #         )
        #     if "GOOGLE_API_KEY" not in os.environ:
        #         os.environ["GOOGLE_API_KEY"] = AppConstants.GEMINI_API_KEY
        #     self.llm = ChatGoogleGenerativeAI(
        #         model=selected_model,
        #         temperature=0,
        #     )
        # else:
        #     selected_model = llm_model or AppConstants.LLM_MODEL
        #     if "OPENAI_API_KEY" not in os.environ:
        #         os.environ["OPENAI_API_KEY"] = AppConstants.OPENAI_API_KEY
        #     self.llm = OpenAI(model_name=selected_model, temperature=0, max_tokens=500)
        
        # if "GOOGLE_API_KEY" not in os.environ:
        #         os.environ["GOOGLE_API_KEY"] = AppConstants.GEMINI_API_KEY


        # self.llm = ChatGoogleGenerativeAI(model=AppConstants.GEMINI_MODEL, temperature=0)
        
        





        if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = AppConstants.OPENAI_API_KEY

        self.llm = ChatOpenAI(
            model=AppConstants.LLM_MODEL,
            temperature=0,
            max_tokens=self._MAX_RESPONSE_TOKENS,
        )


        self.prompt_template_All_VehicleSummary = PromptTemplate(
    template=(
        "You are a helpful chatbot assistant. Use ONLY the provided context to answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Extract and list vehicle summary details from the context, organized vehicle-wise.\n\n"
        "Output exactly one section per unique vehicle number. Do not repeat the same vehicle section.\n"
        "Return a clean, presentable response in this format:\n"
        "For each vehicle in the context:\n\n"
        "**Vehicle: [Vehicle NO] **\n"
        "Never reveal RESELLER_ID, CUSTOMER_ID, ORG_ID, or DEALER_ID.\n"
        "If these fields appear in context, omit them completely from output.\n"
        "1) Summary: one short paragraph.\n"
        "2) Summary points: List all the parameters values.\n"
    ),
    input_variables=["context", "question"],
)

     

        # RAG prompt
        self. prompt_template_RAG = PromptTemplate(
            template=(
                "You are a helpful chatbot assistant. Use ONLY the provided context to answer.\n\n"
                "Conversation history:\n{chat_history}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Return a clean, presentable response in this format:\n"
                "1) Summary: one short paragraph.\n"
                "2) Key points: 2-5 bullet points.\n"
                "3) Data used: short line mentioning important values from context.\n\n"
                "If the context does not contain the answer, say: "
                "\"I could not find this in the provided data.\""
                # chat_history
            ),
            input_variables=["context", "question"],
        )
        self.prompt_template_VehicleDetail = PromptTemplate(
            template=(
                "You are a helpful chatbot assistant. Use ONLY the provided context to answer.\n\n"
                # "Conversation history:\n{chat_history}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "This is a vehicle detail lookup task.\n"
                "Return vehicle identity/details from context only.\n"
                
                "Focus on fields like VEHICLE_NO if present.\n"
                "Hide VEHICLE_ID, MODEL or any other details if they appear in context.\n"
                "Do not invent any value.\n"
                "Do not include RESELLER_ID, CUSTOMER_ID, ORG_ID, or DEALER_ID in output.\n\n"
                "Return a clean response in this format:\n"
                "1) Matched Vehicles: list each matched vehicle on a new line as\n"
                "   - VehicleNo: <value>, VehicleId: <value>, Model: <value>\n"
                "2) Notes: short line for any missing fields as N/A.\n\n"
                "If the context does not contain the answer, say: "
                "\"I could not find this in the provided data.\""
            ),
            input_variables=["context", "question"],
        )
    
        

        self.prompt_template_General = PromptTemplate(
    template=(
        "You are a helpful chatbot assistant.\n"
        "Give only simple and direct answers.\n"
        "Respond ONLY to questions related to vehicles (cars, bikes, trucks, EV, fuel, GPS, telematics, mileage, engine, etc.).\n"
        "Do NOT prefix your answer with words like 'Answer:', 'Response:', etc.\n"
        "If the question is NOT related to vehicles, reply only with:\n"
        "\"This question is not related to vehicles. I cannot answer it.\"\n\n"
        "Conversation history:\n{chat_history}\n\n"

        "Question: {question}\n"
    ),
    input_variables=["chat_history", "question"],
)





        self.prompt_template_query_status = PromptTemplate(
    template=(
        "Question: {question}\n"
        "Analyse the question and Give simple answers only, generic or vehiclesummary-all or vehiclesummary-notall or none"
        "Condition: If the question is generic related to vehicles answer: generic , if the question is related to vehicle summary of all vehicles answer: vehiclesummary-all , if the question is related to specific vehicle summary or particular vehicle summary answer: vehiclesummary-notall,if the question is not related to all the previous given conditions answer: none\n"
    ),
    input_variables=[ "question"],
)

    @classmethod
    def _get_or_create_history(cls, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in cls._chat_histories:
            cls._chat_histories[session_id] = InMemoryChatMessageHistory()
        return cls._chat_histories[session_id]

    @staticmethod
    def _format_recent_history(
        history: InMemoryChatMessageHistory,
        max_messages: int = 6,
        max_chars_per_message: int = 1200,
    ) -> str:
        messages = history.messages[-max_messages:]
        if not messages:
            return "No previous conversation."

        lines: List[str] = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            content = str(msg.content)
            if len(content) > max_chars_per_message:
                content = f"{content[:max_chars_per_message]}..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _llm_response_to_text(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()

        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                return " ".join(text_parts).strip()

        return str(response).strip()

    @staticmethod
    def _normalize_query_label(label: str) -> str:
        normalized = str(label or "").strip().lower()
        allowed = {"generic", "vehiclesummary-all", "vehicledetail","vehiclesummary-notall", "none"}
        if normalized in allowed:
            return normalized
        return "none"

    @staticmethod
    def _response_to_history_text(resp: Any) -> str:
        if isinstance(resp, dict):
            if "response" in resp and resp.get("response") is not None:
                return RAGEngine._response_to_history_text(resp.get("response"))
            if "answer" in resp and resp.get("answer") is not None:
                return str(resp.get("answer"))
        return str(resp or "").strip()

    @classmethod
    def append_history_turn(cls, session_id: str, user_text: str, assistant_text: str) -> None:
        history = cls._get_or_create_history(session_id or "default")
        history.add_message(HumanMessage(content=str(user_text or "")))
        history.add_message(AIMessage(content=str(assistant_text or "")))

    def get_or_create_collection(self, collection_name: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )
    

    def _get_records_by_metadata(
        self,
        collection_name: str,
        where: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> Dict[str, Sequence[Any]]:
        vectordb = self.get_or_create_collection(collection_name)
        kwargs: Dict[str, Any] = {"where": where}
        if limit is not None:
            kwargs["limit"] = limit
        return vectordb.get(**kwargs)

    def get_vehicle_id_by_name(
        self,
        vehicle_name: str,
        vehicleid_collection: str,
        vehicle_name_key: str = "VEHICLE_NO",
        vehicle_id_key: str = "VEHICLE_ID",
    ) -> Optional[str]:
        records = self._get_records_by_metadata(
            collection_name=vehicleid_collection,
            where={vehicle_name_key: vehicle_name},
            limit=1,
        )
        metadatas = records.get("metadatas") or []
        if not metadatas:
            return None
        vehicle_id = metadatas[0].get(vehicle_id_key)
        if vehicle_id is None:
            return None
        return str(vehicle_id)

    @staticmethod
    def _extract_vehicle_name_from_query(query: str) -> str:
        vehicle_names = RAGEngine._extract_vehicle_names_from_text(query)
        if vehicle_names:
            return vehicle_names[0]
        return query.strip()

    @staticmethod
    def _extract_vehicle_names_from_text(text: str) -> List[str]:
        pattern = r"[A-Za-z]{2}\d{1,2}[A-Za-z]{1,3}\d{1,4}"
        normalized = re.sub(r"\s+", "", str(text or "").upper())
        matches = re.findall(pattern, normalized)
        seen = set()
        result: List[str] = []
        for match in matches:
            if match in seen:
                continue
            seen.add(match)
            result.append(match)
        return result

    @staticmethod
    def _is_referential_vehicle_query(query: str) -> bool:
        normalized = str(query or "").lower()
        markers = [
            "this vehicle",
            "that vehicle",
            "same vehicle",
            "previous vehicle",
            "above vehicle",
            "that one",
            "this one",
            "for vehicle",
        ]
        return any(marker in normalized for marker in markers)

    def _resolve_vehicle_name_from_history(
        self,
        query: str,
        history: InMemoryChatMessageHistory,
    ) -> Optional[str]:
        explicit_vehicle_names = self._extract_vehicle_names_from_text(query)
        if explicit_vehicle_names:
            return explicit_vehicle_names[0]
        if not self._is_referential_vehicle_query(query):
            return None

        for msg in reversed(history.messages):
            content = getattr(msg, "content", "")
            vehicle_names = self._extract_vehicle_names_from_text(str(content))
            if vehicle_names:
                return vehicle_names[0]
        return None

    @staticmethod
    def _is_general_vehicle_query(query: str) -> bool:
        if RAGEngine._extract_vehicle_names_from_text(query):
            return False
        if RAGEngine._is_referential_vehicle_query(query):
            return False

        normalized = query.lower()
        general_markers = [
            "vehicle summary",
            "active vehicles",
            "vehicles",
            "vehicle-wise summary",
            "vehicles summary",
            "all vehicle",
            "all vehicles",
            "every vehicle",
            "fleet",
            "overall",
            "across vehicles",
            "general summary",
        ]
        return any(marker in normalized for marker in general_markers)

    
    @staticmethod
    def get_summary_month_from_query(query: str) -> Optional[str]:
        normalized = str(query or "").lower()
        month_aliases = {
            "jan": "jan",
            "january": "jan",
            "feb": "feb",
            "february": "feb",
            "mar": "mar",
            "march": "mar",
            "apr": "apr",
            "april": "apr",
            "may": "may",
            "jun": "jun",
            "june": "jun",
            "jul": "jul",
            "july": "jul",
            "aug": "aug",
            "august": "aug",
            "sep": "sep",
            "sept": "sep",
            "september": "sep",
            "oct": "oct",
            "october": "oct",
            "nov": "nov",
            "november": "nov",
            "dec": "dec",
            "december": "dec",
        }

        for token, month_value in month_aliases.items():
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return month_value
        return None






      
    @staticmethod
    def _is_active_vehicle_count_query(query: str) -> bool:
        normalized = query.lower()
        has_count_intent = any(
            marker in normalized for marker in ["how many", "count", "number of", "total"]
        )
        has_vehicle_intent = "vehicle" in normalized
        has_active_intent = "active" in normalized
        return has_count_intent and has_vehicle_intent and has_active_intent

    def _count_unique_vehicles(
        self,
        metadatas: List[Dict[str, Any]],
        vehicle_name_key: str,
        vehicle_id_key: str,
    ) -> int:
        unique_vehicles = set()
        for metadata in metadatas:
            metadata = metadata or {}
            value = self._pick_metadata_value(
                metadata,
                [
                    vehicle_name_key,
                    "VEHICLE_NO",
                    "vehicleNo",
                    "vehicle_name",
                    "vehicleno",
                    vehicle_id_key,
                    "VEHICLE_ID",
                    "vehicleId",
                    "vehicleid",
                ],
            )
            if value:
                unique_vehicles.add(str(value))
        return len(unique_vehicles)

    @staticmethod
    def _build_where_filter(
        filters: Dict[str, Any],
        operator: str = "$and",
    ) -> Optional[Dict[str, Any]]:
        cleaned = {k: v for k, v in filters.items() if v is not None}
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned
        if operator not in {"$and", "$or"}:
            operator = "$and"
        return {operator: [{k: v} for k, v in cleaned.items()]}

    @staticmethod
    def _get_claim_value(claims: Dict[str, Any], keys: List[str]) -> Any:
        for key in keys:
            if key in claims and claims.get(key) not in (None, ""):
                return claims.get(key)
        lowered = {str(k).lower(): v for k, v in claims.items()}
        for key in keys:
            value = lowered.get(key.lower())
            if value not in (None, ""):
                return value
        return None

    def get_userrole(self, claims: Dict[str, Any]) -> Optional[str]:
        role_id = self._get_claim_value(
            claims or {},
            ["roleId", "role_id", "ROLE_ID", "roleid"],
        )
        if role_id in (None, ""):
            return self._get_claim_value(
                claims or {},
                ["userRole", "user_role", "USER_ROLE", "userrole"],
            )

        role_id = self._coerce_filter_value(role_id)
        role_desc_keys = [
            "ROLE_DESC",
            "roleDesc",
            "role_desc",
            "USER_ROLE",
            "userRole",
            "role",
        ]

        role_id_variants = [role_id]
        if isinstance(role_id, int):
            role_id_variants.append(str(role_id))
        elif isinstance(role_id, str) and role_id.isdigit():
            role_id_variants.append(int(role_id))

        where_keys = ["ID", "id", "ROLE_ID", "roleId", "roleid"]
        for where_key in where_keys:
            for role_id_value in role_id_variants:
                try:
                    records = self._get_records_by_metadata(
                        collection_name="role_data_collection",
                        where={where_key: role_id_value},
                        limit=1,
                    )
                except Exception:
                    continue

                metadatas = records.get("metadatas") or []
                if not metadatas:
                    continue

                metadata = metadatas[0] if isinstance(metadatas[0], dict) else {}
                role_desc = self._pick_metadata_value(metadata, role_desc_keys)
                if role_desc:
                    return role_desc

        return self._get_claim_value(
            claims or {},
            ["userRole", "user_role", "USER_ROLE", "userrole"],
        )

    # def _build_claims_where_new(self, claims: Dict[str, Any]) -> Optional[Dict[str, Any]]:


    #     key_mapping = {
    #     "resellerId": "reseller_id",
    #     "reseller_id": "reseller_id",
    #     "RESELLER_ID": "reseller_id",
    #     "resellerid": "reseller_id",

    #     "customerId": "customer_id",
    #     "customer_id": "customer_id",
    #     "CUSTOMER_ID": "customer_id",
    #     "customerid": "customer_id",

    #     "orgId": "org_id",
    #     "org_id": "org_id",
    #     "ORG_ID": "org_id",
    #     "orgid": "org_id",

    #     "dealerId": "dealer_id",
    #     "dealer_id": "dealer_id",
    #     "DEALER_ID": "dealer_id",
    #     "dealerid": "dealer_id",
    # }

    #     and_conditions: List[Dict[str, Any]] = []

    #     for claim_key, metadata_key in key_mapping.items():
    #         raw_value = claims.get(claim_key)
    #         if raw_value in (None, ""):
    #             continue

    #         coerced_value = self._coerce_filter_value(raw_value)

    #     # Avoid duplicate conditions
    #         if not any(metadata_key in cond for cond in and_conditions):
    #             and_conditions.append({metadata_key: coerced_value})

    #     if not and_conditions:
    #         return None

    #     if len(and_conditions) == 1:
    #         return and_conditions[0]

    #     return {"$and": and_conditions}



    def _build_claims_where_new(self, claims: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build strict AND-based metadata filter using standardized UPPERCASE keys.
        """

        key_mapping = {
        "resellerId": "RESELLER_ID",
        "reseller_id": "RESELLER_ID",
        "RESELLER_ID": "RESELLER_ID",

        "customerId": "CUSTOMER_ID",
        "customer_id": "CUSTOMER_ID",
        "CUSTOMER_ID": "CUSTOMER_ID",

        "orgId": "ORG_ID",
        "org_id": "ORG_ID",
        "ORG_ID": "ORG_ID",

        "dealerId": "DEALER_ID",
        "dealer_id": "DEALER_ID",
        "DEALER_ID": "DEALER_ID",
    }

        normalized = {}

        # Normalize claims to single uppercase metadata format
        for claim_key, metadata_key in key_mapping.items():
            value = claims.get(claim_key)
            if value in (None, ""):
                continue

            coerced_value = self._coerce_filter_value(value)

        # Only set once per metadata key
            if metadata_key not in normalized:
                 normalized[metadata_key] = coerced_value

        if not normalized:
            return None

        and_conditions = [{k: v} for k, v in normalized.items()]

        if len(and_conditions) == 1:
            return and_conditions[0]

        return {"$and": and_conditions}
   

    def _build_claims_where(self, claims: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        claim_specs = [
            (["resellerId", "reseller_id", "RESELLER_ID", "resellerid"], ["RESELLER_ID", "resellerid", "resellerId"]),
            (["customerId", "customer_id", "CUSTOMER_ID", "customerid"], ["CUSTOMER_ID", "customerid", "customerId"]),
            (["orgId", "org_id", "ORG_ID", "orgid"], ["ORG_ID", "orgid", "orgId"]),
            (["dealerId", "dealer_id", "DEALER_ID", "dealerid"], ["DEALER_ID", "dealerid", "dealerId"]),
        ]

        or_conditions: List[Dict[str, Any]] = []
        for claim_keys, metadata_keys in claim_specs:
            raw_value = self._get_claim_value(claims, claim_keys)
            if raw_value in (None, ""):
                continue
            coerced_value = self._coerce_filter_value(raw_value)
            value_variants = [coerced_value]
            if isinstance(coerced_value, int):
                value_variants.append(str(coerced_value))
            elif isinstance(coerced_value, str) and coerced_value.isdigit():
                value_variants.append(int(coerced_value))

            seen = set()
            for metadata_key in metadata_keys:
                for value in value_variants:
                    signature = (metadata_key, str(value))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    or_conditions.append({metadata_key: value})

        if not or_conditions:
            return None
        if len(or_conditions) == 1:
            return or_conditions[0]
        return {"$or": or_conditions}

    @staticmethod
    def _coerce_filter_value(value: Any) -> Any:
        # Chroma metadata matching is type-sensitive; coerce numeric-like claims.
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            return stripped
        return value

    @staticmethod
    def _pick_metadata_value(metadata: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)

        lowered = {str(k).lower(): v for k, v in metadata.items()}
        for key in keys:
            value = lowered.get(key.lower())
            if value not in (None, ""):
                return str(value)
        return None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough approximation for English text to prevent prompt overflow.
        return max(1, len(text) // 4)

    def _count_tokens(self, text: str) -> int:
        content = str(text or "")
        if not content:
            return 0
        if tiktoken is not None:
            try:
                model_name = getattr(self.llm, "model_name", "") or ""
                encoding = (
                    tiktoken.encoding_for_model(model_name)
                    if model_name
                    else tiktoken.get_encoding("cl100k_base")
                )
                return len(encoding.encode(content))
            except Exception:
                pass
        return self._estimate_tokens(content)

    def _print_token_usage(self, prompt_text: str, output_text: str, route_name: str) -> None:
        input_tokens = self._count_tokens(prompt_text)
        output_tokens = self._count_tokens(output_text)
        logger.info(
            "[%s] input_tokens=%s output_tokens=%s",
            route_name,
            input_tokens,
            output_tokens,
        )
        print(f"[{route_name}] input_tokens={input_tokens} output_tokens={output_tokens}")

    def _build_bounded_context(
        self,
        docs: List[str],
        token_budget: int,
    ) -> str:
        if not docs:
            return ""

        pieces: List[str] = []
        used_tokens = 0

        for doc in docs:
            text = str(doc or "").strip()
            if not text:
                continue
            part_tokens = self._estimate_tokens(text) + 2
            if used_tokens + part_tokens > token_budget:
                break
            pieces.append(text)
            used_tokens += part_tokens

        return "\n\n---\n\n".join(pieces)

    def _build_vehicle_grouped_context_docs(
        self,
        results: List[Dict[str, Any]],
    ) -> List[str]:
        grouped_docs: Dict[str, List[str]] = {}
        seen_per_vehicle: Dict[str, set] = {}

        for item in results:
            document = self._normalize_text(item.get("document"))
            if not document:
                continue

            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            vehicle_no = (
                item.get("vehicleNo")
                or metadata.get("VEHICLE_NO")
                or metadata.get("vehicleNo")
                or "UNKNOWN"
            )
            vehicle_no = str(vehicle_no).strip() or "UNKNOWN"

            if vehicle_no not in grouped_docs:
                grouped_docs[vehicle_no] = []
                seen_per_vehicle[vehicle_no] = set()

            if document in seen_per_vehicle[vehicle_no]:
                continue

            seen_per_vehicle[vehicle_no].add(document)
            grouped_docs[vehicle_no].append(document)

        context_docs: List[str] = []
        for vehicle_no, docs in grouped_docs.items():
            merged_document = "\n".join(docs)
            context_docs.append(f"VehicleNo: {vehicle_no}\nDocument:\n{merged_document}")

        return context_docs

    def _format_all_vehicle_response(
        self,
        results: List[Dict[str, Any]],
        vehicle_name_key: str,
        vehicle_id_key: str,
    ) -> str:
        if not results:
            return "I could not find this in the provided data."

        blocks: List[str] = []
        for index, item in enumerate(results, start=1):
            metadata = item.get("metadata") or {}
            document_text = self._normalize_text(item.get("document", ""))

            vehicle_label = self._pick_metadata_value(
                metadata,
                [
                    vehicle_name_key,
                    "VEHICLE_NO",
                    "vehicleNo",
                    "vehicle_name",
                    "vehicleno",
                    vehicle_id_key,
                    "VEHICLE_ID",
                    "vehicleId",
                    "vehicleid",
                ],
            ) or f"UNKNOWN_{index}"

            date_value = self._pick_metadata_value(
                metadata, ["DATE", "date", "MSGDATE", "msgdate"]
            )
            distance_value = self._pick_metadata_value(
                metadata,
                [
                    "DISTANCE",
                    "distance",
                    "TOTAL_DISTANCE",
                    "total_distance",
                    "KM",
                    "km",
                ],
            )
            start_time = self._pick_metadata_value(
                metadata, ["START_TIME", "start_time", "FROM_TIME", "from_time", "start"]
            )
            end_time = self._pick_metadata_value(
                metadata, ["END_TIME", "end_time", "TO_TIME", "to_time", "end"]
            )

            summary_text = document_text or "No summary text found in the record."
            if len(summary_text) > 240:
                summary_text = f"{summary_text[:237]}..."

            key_points: List[str] = []
            if date_value:
                key_points.append(f"- Date: {date_value}")
            if distance_value:
                key_points.append(f"- Distance: {distance_value}")
            if start_time or end_time:
                key_points.append(
                    f"- Time: {(start_time or 'N/A')} to {(end_time or 'N/A')}"
                )

            if not key_points and metadata:
                for key, value in list(metadata.items())[:3]:
                    if value not in (None, ""):
                        key_points.append(f"- {key}: {value}")

            data_used_values = [v for v in [distance_value, start_time, end_time] if v]
            if data_used_values:
                data_used = ", ".join(data_used_values)
            elif document_text:
                data_used = summary_text
            else:
                data_used = "N/A"

            key_points_text = "\n".join([f"   {point}" for point in key_points]) or "   - N/A"

            block = (
                f"---\n\n"
                f"**Vehicle: {vehicle_label}**\n"
                f"1) Summary: {summary_text}\n"
                f"2) Key points:\n{key_points_text}\n"
                f"3) Data used: {data_used}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def get_vehicle_summary_by_name(
        self,
        query: str,
        vehicleid_collection: str,
        vehicle_name_key: str = "VEHICLE_NO",
        vehicle_id_key: str = "VEHICLE_ID",
        k: int = 5,
        session_id: str = "default",
        claims: Optional[Dict[str, Any]] = None,
        persist_history: bool = True,
    ) -> Dict[str, Any]:
        


        session_id = session_id or "default"
        claims = claims or {}
        history = self._get_or_create_history(session_id)
        
        resolved_vehicle_name = self._resolve_vehicle_name_from_history(query, history)
        is_general_query = self._is_general_vehicle_query(query)
        summary_month = self.get_summary_month_from_query(query)



    #    vehiclesummary_feb_collection_v3
        if summary_month is None:
             summary_month = datetime.now().strftime("%b").lower()
        summary_collection = f"vehiclesummary_{summary_month}_collection_v3"
    
        vehicle_name = resolved_vehicle_name or self._extract_vehicle_name_from_query(query)
        if resolved_vehicle_name:
            is_general_query = False
        vehicle_id: Optional[str] = None

        if not is_general_query:
            vehicle_id = self.get_vehicle_id_by_name(
                vehicle_name=vehicle_name,
                vehicleid_collection=vehicleid_collection,
                vehicle_name_key=vehicle_name_key,
                vehicle_id_key=vehicle_id_key,
            )

        if vehicle_id is None and not is_general_query:
            vehicleid_db = self.get_or_create_collection(vehicleid_collection)
            similar_vehicle_docs = vehicleid_db.similarity_search(query=query, k=1)
            if similar_vehicle_docs:
                matched_meta = similar_vehicle_docs[0].metadata or {}
                matched_vehicle_name = matched_meta.get(vehicle_name_key)
                matched_vehicle_id = matched_meta.get(vehicle_id_key)
                if matched_vehicle_name:
                    vehicle_name = str(matched_vehicle_name)
                if matched_vehicle_id is not None:
                    vehicle_id = str(matched_vehicle_id)

        summary_db = self.get_or_create_collection(summary_collection)
        results: List[Dict[str, Any]] = []
        claims_where = self._build_claims_where(claims)
        logger.info("JWT claims received: %s", claims)
        logger.info("Resolved claims_where filter: %s", claims_where)


        # claims_filter = {
        #     "RESELLER_ID": 116606,
        #     "CUSTOMER_ID": 116657,
        #     "ORG_ID": 116659,
        #     "DEALER_ID": 116652,
        # }
         

        # claims_filter = {
        #     "RESELLER_ID": 116606,
        #     "CUSTOMER_ID": 116657,
        #     "ORG_ID": 116659,
        #     "DEALER_ID": 116652,
        #     "RESELLER_ID": self._coerce_filter_value(claims.get("resellerId")),
        #     "CUSTOMER_ID": self._coerce_filter_value(claims.get("customerId")),
        #     "ORG_ID": self._coerce_filter_value(claims.get("orgId")),
        #     "DEALER_ID": self._coerce_filter_value(claims.get("dealerId")),
        # }
         

        if self._is_active_vehicle_count_query(query):
            records = summary_db.get(where=claims_where) if claims_where else summary_db.get()
            metadatas = records.get("metadatas") or []
            active_vehicle_count = self._count_unique_vehicles(
                metadatas=metadatas,
                vehicle_name_key=vehicle_name_key,
                vehicle_id_key=vehicle_id_key,
            )
            answer_text = f"There are {active_vehicle_count} active vehicles."
            if persist_history:
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=answer_text))
            return {
                "query": query,
                "vehicle_name": "ALL_VEHICLES",
                "vehicle_id": None,
                "session_id": session_id,
                "response": answer_text,
                "active_vehicle_count": active_vehicle_count,
                "is_general_query": True,
            }

        if vehicle_id is None and is_general_query:
            vehicle_name = "ALL_VEHICLES"
            if claims_where:
                all_records = summary_db.get(where=claims_where)
            else:
                all_records = summary_db.get()
            ids = all_records.get("ids") or []
            docs = all_records.get("documents") or []
            metas = all_records.get("metadatas") or []
            result_count = min(len(ids), len(docs), len(metas))

            for i in range(result_count):
                vehicle_no = (
                    metas[i].get("VEHICLE_NO")
                    if isinstance(metas[i], dict)
                    else None
                )
                results.append(
                    {
                        "document": docs[i],
                 
                    }
                )
        elif vehicle_id is None:
            answer_text = "I could not find this in the provided data."
            if persist_history:
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=answer_text))
            return {
                "query": query,
                "vehicle_name": vehicle_name,
                "vehicle_id": None,
                "session_id": session_id,
                "response": answer_text,
                "results": [],
            }
        else:
            string_where: Dict[str, Any] = {vehicle_id_key: vehicle_id}
            if claims_where:
                string_where = {"$and": [{vehicle_id_key: vehicle_id}, claims_where]}

            int_where: Optional[Dict[str, Any]] = None
            if vehicle_id.isdigit():
                int_where = {vehicle_id_key: int(vehicle_id)}
                if claims_where:
                    int_where = {"$and": [{vehicle_id_key: int(vehicle_id)}, claims_where]}

            summary_records = self._get_records_by_metadata(
                collection_name=summary_collection,
                where=string_where,
            )
            if not (summary_records.get("ids") or []) and vehicle_id.isdigit():
                summary_records = self._get_records_by_metadata(
                    collection_name=summary_collection,
                    where=int_where,
                )

            ids = summary_records.get("ids") or []
            docs = summary_records.get("documents") or []
            metas = summary_records.get("metadatas") or []
            result_count = min(len(ids), len(docs), len(metas))

            for i in range(result_count):
                results.append(
                    {
                        "document": docs[i],
                    }
                )

        if not results:
            answer_text = "I could not find this in the provided data."
            if persist_history:
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=answer_text))
            return {
                "query": query,
                "vehicle_name": vehicle_name,
                "vehicle_id": vehicle_id,
                "session_id": session_id,
                "response": answer_text,
                "results": [],
            }

        # if is_general_query and vehicle_id is None:
        #     answer_text = self._format_all_vehicle_response(
        #         results=results,
        #         vehicle_name_key=vehicle_name_key,
        #         vehicle_id_key=vehicle_id_key,
        #     )
        #     history.add_message(HumanMessage(content=query))
        #     history.add_message(AIMessage(content=answer_text))
        #     return {
        #         "query": query,
        #         "vehicle_name": vehicle_name,
        #         "vehicle_id": vehicle_id,
        #         "session_id": session_id,
        #         "response": answer_text,
        #         # "results": results,
        #         "is_general_query": is_general_query,
        #     }

        context_docs = self._build_vehicle_grouped_context_docs(results)
        # context = self._build_bounded_context(
        #     context_docs,
        #     token_budget=self._CONTEXT_BUDGET_TOKENS,
        # )
        if not context_docs:
            answer_text = "I could not find this in the provided data."
            if persist_history:
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=answer_text))
            return {
                "query": query,
                "vehicle_name": vehicle_name,
                "vehicle_id": vehicle_id,
                "session_id": session_id,
                "response": answer_text,
                "results": [],
                "is_general_query": is_general_query,
            }

        prompt = self.prompt_template_All_VehicleSummary.format(
            chat_history=self._format_recent_history(history),
            context=context_docs,
            question=query
        )

        answer = self.llm.invoke(prompt)
        answer_text = self._llm_response_to_text(answer)
        self._print_token_usage(prompt, answer_text, "vehiclesummary")
        if persist_history:
            history.add_message(HumanMessage(content=query))
            history.add_message(AIMessage(content=answer_text))

        return {
            "query": query,
            "vehicle_name": vehicle_name,
            "vehicle_id": vehicle_id,
            "session_id": session_id,
            "response": answer_text,
            "results": results,
            "is_general_query": is_general_query,
        }

   
    def answer_question(self, req: AskRequest):
        session_id = req.session_id or "default"
        history = self._get_or_create_history(session_id)

        filter_dict = req.filter.dict(exclude_none=True) if req.filter else None
        if not filter_dict:
            filter_dict = None

        filter_dict1 = {
    # "source": {"$eq": "string"},
    # "topic": {"$eq": "string"},
    # "msgtimestamp": {"$eq": 0},
     "msgdate": "2025-12-11",
    # "orgid": {"$eq": 0},
    # "resellerid": {"$eq": 0}
    # "customerid": {"$eq": 0},
    # "dealerid": {"$eq": 0},
    # "regionid": {"$eq": 0},
    # "createddate": "2025-06-03 08:55:30"
}


        docs = self.vectorstore.similarity_search(
        query=req.query,
        k=req.k,
        filter=filter_dict1)
        

        logger.info(f"docs ---> {docs}")
        context_texts = [d.page_content for d in docs]
        sources = [d.metadata for d in docs]

        context = self._build_bounded_context(
            context_texts,
            token_budget=self._CONTEXT_BUDGET_TOKENS,
        )
        logger.info(f"context ---> {context}")
        if not context:
            return {
                "answer": "No relevant context found in vector store.",
                "sources": [],
                "chunks": []
            }
             

        prompt = self.prompt_template_All_VehicleSummary.format(
            chat_history=self._format_recent_history(history),
            context=context,
            question=req.query
        )

        response = self.llm.invoke(prompt)
        answer_text = self._llm_response_to_text(response)
        self._print_token_usage(prompt, answer_text, "ask_RAG")
        history.add_message(HumanMessage(content=req.query))
        history.add_message(AIMessage(content=answer_text))
        # self.conversation_history.append(HumanMessage(content=question))

    



    #     messages = [
    #     self.system_message,
    #     *self.conversation_history,
    #     HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    # ]

    #     result = self.llm.invoke(messages)

    # Save AI response
        # self.conversation_history.append(AIMessage(content=result))

        # return {
        #       "answer": response,
        #       "sources": sources,
        #       "chunks": context_texts}

        return {                         
              "answer": answer_text,
              "session_id": session_id,
              "sources": sources,
              "chunks": context_texts}


    def persist(self):
        self.vectorstore.persist()


    def answer_vehicle(
        self,
        query: str,
        session_id: str = "default",
        persist_history: bool = True,
    ) -> Dict[str, Any]:
        try:
            session_id = session_id or "default"
            history = self._get_or_create_history(session_id)

            prompt_general = self.prompt_template_General.format(
                chat_history=self._format_recent_history(history),
                question=query,
            )

            logger.info("prompt_general -----> %s", prompt_general)

            response = self.llm.invoke(prompt_general)
            answer_text = self._llm_response_to_text(response)
            self._print_token_usage(prompt_general, answer_text, "ask_Vehicles")
            if persist_history:
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=answer_text))

            return {"answer": answer_text, "session_id": session_id}
        except Exception as e:
            logger.error(f"Error in answer_vehicle: {e}")
            return {
                "answer": "An error occurred while processing your request.",
                "session_id": session_id or "default",
            }
        

    def get_vehicle_detail(
        self,
        query: str,
        claims: Optional[Dict[str, Any]] = None,
        session_id: str = "default",
        persist_history: bool = True,
        vehicle_detail_collection: str = "vehicle_data_collection5",
        k: int = 20,
    ) -> Dict[str, Any]:
        session_id = session_id or "default"
        history = self._get_or_create_history(session_id)
        claims = claims or {}
        claims_where = self._build_claims_where_new(claims)

        logger.info("get_vehicle_detail claims: %s", claims)
        logger.info("get_vehicle_detail claims_where: %s", claims_where)

        detail_db = self.get_or_create_collection(vehicle_detail_collection)
        explicit_vehicle_names = self._extract_vehicle_names_from_text(query)
        results: List[Dict[str, Any]] = []

        if explicit_vehicle_names:
            vehicle_no = explicit_vehicle_names[0]
            where_filter: Dict[str, Any] = {"VEHICLE_NO": vehicle_no}
            if claims_where:
                where_filter = {"$and": [{"VEHICLE_NO": vehicle_no}, claims_where]}
            records = detail_db.get(where=where_filter)
            ids = records.get("ids") or []
            docs = records.get("documents") or []
            metas = records.get("metadatas") or []
            result_count = min(len(ids), len(docs), len(metas))
            for i in range(result_count):
                results.append(
                    {
                        "id": ids[i],
                        "document": docs[i],
                        "metadata": metas[i] if isinstance(metas[i], dict) else {},
                    }
                )
        else:
            # If query has no explicit vehicle number, return records by claim scope.
            if claims_where:
                records = detail_db.get(where=claims_where)
                ids = records.get("ids") or []
                docs = records.get("documents") or []
                metas = records.get("metadatas") or []
                result_count = min(len(ids), len(docs), len(metas))
                for i in range(result_count):
                    results.append(
                        {
                            "id": ids[i],
                            "document": docs[i],
                            "metadata": metas[i] if isinstance(metas[i], dict) else {},
                        }
                    )
            else:
                similar_docs = detail_db.similarity_search(query=query, k=k)
                for doc in similar_docs:
                    results.append(
                        {
                            "id": None,
                            "document": getattr(doc, "page_content", ""),
                            "metadata": getattr(doc, "metadata", {}) or {},
                        }
                    )

        context_docs = self._build_vehicle_grouped_context_docs(results)
        context = self._build_bounded_context(
            context_docs,
            token_budget=self._CONTEXT_BUDGET_TOKENS,
        )
        if not context:
            answer_text = "I could not find this in the provided data."
            if persist_history:
                history.add_message(HumanMessage(content=query))
                history.add_message(AIMessage(content=answer_text))
            return {
                "query": query,
                "session_id": session_id,
                "response": answer_text,
                "results": [],
            }

        prompt = self.prompt_template_VehicleDetail.format(
            # chat_history=self._format_recent_history(history),
            context=context,
            question=query
        )

        print(prompt)

        answer = self.llm.invoke(prompt)
        answer_text = self._llm_response_to_text(answer)
        self._print_token_usage(prompt, answer_text, "vehiclesummary")
        if persist_history:
            history.add_message(HumanMessage(content=query))
            history.add_message(AIMessage(content=answer_text))

        return {
            "query": query,
            "session_id": session_id,
            "response": answer_text,
            "results": results,
        }



    def query_status_check(
        self,
        query: str,
        session_id: str,
        use_history_for_classification: bool = True,
    ) -> str:
        try:
            session_id = session_id or "default"
            history = self._get_or_create_history(session_id)
            history_text = (
                self._format_recent_history(history)
                if use_history_for_classification
                else "No previous conversation."
            )

            classifier_prompt = (
                "You are a strict query classifier.\n"
                "Classify the CURRENT QUERY into exactly one label from this list:\n"
                "generic, vehicledetail, vehiclesummary-all, vehiclesummary-notall, none\n\n"
                "Decision rules (apply in this order):\n"
                "1) If the user asks for summary/report/overview/status/health of vehicles:\n"
                "   - Use vehiclesummary-all for all vehicles/fleet/every vehicle.\n"
                "   - Use vehiclesummary-notall for one/some specific vehicles.\n"
                "2) If the user asks for vehicle identity/details (vehicle number, vehicle id, model,\n"
                "   registration, list/find/search/show vehicles) and does NOT ask for summary/report/overview,\n"
                "   return vehicledetail.\n"
                "3) If the user asks a general vehicle-domain question not requesting vehicle lookup/summary,\n"
                "   return generic.\n"
                "4) If unrelated to vehicle domain, return none.\n\n"
                "Important:\n"
                "- Prefer vehicledetail when query is about identifying or listing vehicles.\n"
                "- Use chat history only for disambiguation; prioritize CURRENT QUERY.\n"
                "- Return only the label text, no explanation or punctuation.\n\n"
                f"Chat history:\n{history_text}\n\n"
                f"Current query:\n{query}\n"
            )

            response = self.llm.invoke(classifier_prompt)
          
            answer_text = self._llm_response_to_text(response)
            label_match = re.search(
                r"\b(generic|vehicledetail|vehiclesummary-all|vehiclesummary-notall|none)\b",
                answer_text.strip().lower(),
            )
            if label_match:
                answer_text = label_match.group(1)
            self._print_token_usage(classifier_prompt, answer_text, "query_status_check")
            return self._normalize_query_label(answer_text)
        except Exception as e:
            logger.error(f"Error in query_status_check: {e}")
            return "none"



      
