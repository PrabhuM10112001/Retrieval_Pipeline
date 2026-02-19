# rag_engine.py
import os
import re
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
from langchain_openai import OpenAI

from langchain_core.prompts import PromptTemplate
from ApplicationConstants import AppConstants
from Schemas import AskRequest, AskVehicleRequest
from logger import logger



class RAGEngine:
    _chat_histories:  Dict[str, InMemoryChatMessageHistory] = {}
    def __init__(
        self,
        collection_name: str = "t_defaultcollection",
        persist_directory: str = AppConstants.VECTOR_DB_LOC,
        embedding_model: str = AppConstants.EMBEDDING_MODEL,
        llm_model: str = AppConstants.LLM_MODEL,  
      
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
        if 'OPENAI_API_KEY' not in os.environ:
            os.environ['OPENAI_API_KEY'] = AppConstants.OPEN_API_KEY

    
        self.llm = OpenAI(model_name=llm_model, temperature=0, max_tokens=500)

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

            ),
            input_variables=["chat_history", "context", "question"],
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
  
    @classmethod
    def _get_or_create_history(cls, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in cls._chat_histories:
            cls._chat_histories[session_id] = InMemoryChatMessageHistory()
        return cls._chat_histories[session_id]


    @classmethod
    def _get_or_create_history(cls, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in cls._chat_histories:
            cls._chat_histories[session_id] = InMemoryChatMessageHistory()
        return cls._chat_histories[session_id]

    @staticmethod
    def _format_recent_history(history: InMemoryChatMessageHistory, max_messages: int = 10) -> str:
        messages = history.messages[-max_messages:]
        if not messages:
            return "No previous conversation."

        lines: List[str] = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

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
        vehicle_name_key: str = "vehicleNo",
        vehicle_id_key: str = "vehicleid",
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
        pattern = r"[A-Za-z]{2}\d{1,2}[A-Za-z]{1,3}\d{1,4}"
        match = re.search(pattern, query.replace(" ", ""))
        if match:
            return match.group(0)
        return query.strip()

    @staticmethod
    def _is_general_vehicle_query(query: str) -> bool:
        normalized = query.lower()
        general_markers = [
            "all vehicle",
            "all vehicles",
            "every vehicle",
            "fleet",
            "overall",
            "across vehicles",
            "general summary",
        ]
        return any(marker in normalized for marker in general_markers)

    def get_vehicle_summary_by_name(
        self,
        query: str,
        vehicleid_collection: str,
        summary_collection: str,
        vehicle_name_key: str = "vehicleNo",
        vehicle_id_key: str = "vehicleid",
        k: int = 5,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        session_id = session_id or "default"
        history = self._get_or_create_history(session_id)
        is_general_query = self._is_general_vehicle_query(query)
        vehicle_name = self._extract_vehicle_name_from_query(query)
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

        if vehicle_id is None and is_general_query:
            vehicle_name = "ALL_VEHICLES"
            all_records = summary_db.get()
            ids = all_records.get("ids") or []
            docs = all_records.get("documents") or []
            metas = all_records.get("metadatas") or []
            result_count = min(len(ids), len(docs), len(metas))

            for i in range(result_count):
                results.append(
                    {
                        "id": ids[i],
                        "document": docs[i],
                        "metadata": metas[i],
                    }
                )
        elif vehicle_id is None:
            answer_text = "I could not find this in the provided data."
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
            summary_docs = summary_db.similarity_search(
                query=query,
                k=k,
                filter={vehicle_id_key: vehicle_id},
            )

            if not summary_docs and vehicle_id.isdigit():
                summary_docs = summary_db.similarity_search(
                    query=query,
                    k=k,
                    filter={vehicle_id_key: int(vehicle_id)},
                )

            if summary_docs:
                for doc in summary_docs:
                    results.append(
                        {
                            "id": doc.metadata.get("id"),
                            "document": doc.page_content,
                            "metadata": doc.metadata,
                        }
                    )
            else:
                summary_records = self._get_records_by_metadata(
                    collection_name=summary_collection,
                    where={vehicle_id_key: vehicle_id},
                )
                if not (summary_records.get("ids") or []) and vehicle_id.isdigit():
                    summary_records = self._get_records_by_metadata(
                        collection_name=summary_collection,
                        where={vehicle_id_key: int(vehicle_id)},
                    )

                ids = summary_records.get("ids") or []
                docs = summary_records.get("documents") or []
                metas = summary_records.get("metadatas") or []
                result_count = min(len(ids), len(docs), len(metas))

                for i in range(result_count):
                    results.append(
                        {
                            "id": ids[i],
                            "document": docs[i],
                            "vehicleNo": vehicle_name,
                            "metadata": metas[i],
                        }
                    )

        if not results:
            answer_text = "I could not find this in the provided data."
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

        context = "\n\n---\n\n".join(
            [str(item.get("document", "")) for item in results if item.get("document")]
        )

        prompt = self.prompt_template_RAG.format(
            chat_history=self._format_recent_history(history),
            context=context,
            question=query
        )

        answer = self.llm.invoke(prompt)
        answer_text = answer.strip() if isinstance(answer, str) else str(answer).strip()
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
        filter=filter_dict1
    )
        
        
        




        logger.info(f"docs ---> {docs}")
        context_texts = [d.page_content for d in docs]
        sources = [d.metadata for d in docs]

        context = "\n\n---\n\n".join(context_texts)
        logger.info(f"context ---> {context}")
        if not context:
            return {
                "answer": "No relevant context found in vector store.",
                "sources": [],
                "chunks": []
            }
        
        

        prompt = self.prompt_template_RAG.format(
            chat_history=self._format_recent_history(history),
            context=context,
            question=req.query
        )

        response = self.llm.invoke(prompt)
        answer_text = response.strip() if isinstance(response, str) else str(response).strip()
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


#     def add_json_from_api(self, url: str):

         
#         login_url = AppConstants.OBD_LOGIN_URL
    
#         payload_login = AppConstants.OBD_LOGIN_CREDENTIAL

#         # Send login request
#         loginResponse = requests.post(login_url, json=payload_login, verify=False)
#         loginResponse_JSON = loginResponse.json()

#         print("Login Response:", loginResponse_JSON)

#         payload = {
#     "columnName": "CREATED_DATE",
#     "customerId": "-1",
#     "dealerId": 0,
#     "driverId": "-1",
#     "fleetId": "-1",
#     "loggedinOrgId": 113779,
#     "orderDir": "desc",
#     "orgId": 113779,
#     "orgType": "-1",
#     "orgTypeFilter": "-1",
#     "page": 0,
#     "regionId": 0,
#     "resellerId": 113779,
#     "roleId": 17,
#     "searchText": "",
#     "size": 25,
#     "subResellerId": "-1"
# }

#         access_token = loginResponse_JSON.get("accessToken")

#         headers = {"Authorization": f"Bearer {access_token}","Accept": "application/json"}

#         response = requests.post(url, json=payload, headers=headers, verify=False)
#         response.raise_for_status()

#         json_data = response.json()
        
#         print("List User Response: --->", json_data)
        
#         print("json_string:--->",json_data)
#         # convert json to docs  
#         docs = []

#         def flatten(obj, prefix=""):
#              if isinstance(obj, dict):
#                  for k, v in obj.items():
#                     flatten(v, prefix + k + ".")
#              elif isinstance(obj, list):
#                  for i, v in enumerate(obj):
#                     flatten(v, prefix + str(i) + ".")
#              else:
#                 docs.append(Document(page_content=f"{prefix[:-1]}: {obj}"))

#         flatten(json_data)
#         print(docs)
#         # chunking
#         chunks = self.splitter.split_documents(docs)

#         if not chunks:
#             return {"added": 0, "message": "No chunks created from API JSON"}

#         self.vectorstore.add_documents(chunks)
#         self.vectorstore.persist()

#         return {"added": len(chunks)}


    def persist(self):
        self.vectorstore.persist()


    def answer_vehicle(self, req: AskVehicleRequest):
     try:
        session_id = req.session_id or "default"
        history = self._get_or_create_history(session_id)

        promptGeneral = self.prompt_template_General.format(
            chat_history=self._format_recent_history(history),
            question=req.query
        )

        logger.info(f"promptGeneral -----> {promptGeneral}")

        response = self.llm.invoke(promptGeneral)
        answer_text = response.strip() if isinstance(response, str) else str(response).strip()
        history.add_message(HumanMessage(content=req.query))
        history.add_message(AIMessage(content=answer_text))

        return {"answer": answer_text, "session_id": session_id}
     except Exception as e:
        logger.error(f"Error in answer_vehicle: {e}")
        return {"answer": "An error occurred while processing your request."}

   
      
