import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from logger import logger

if TYPE_CHECKING:
    from rag_engine import RAGEngine


NormalizedSummaryRecord = Dict[str, Any]


class VehicleSummaryAllBatchProcessor:
    _LABEL_PATTERN = re.compile(
        r"\b(generic|vehicledetail|vehiclesummary-all|vehiclesummary-notall|none)\b",
        re.IGNORECASE,
    )

    _MONTH_ALIASES: Dict[str, str] = {
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

    def __init__(
        self,
        rag_engine: Optional["RAGEngine"] = None,
        batch_size: int = 20,
        inter_request_delay_seconds: float = 0.3,
        llm_max_retries: int = 4,
        llm_base_backoff_seconds: float = 2.0,
        llm_max_backoff_seconds: float = 45.0,
    ):
        if rag_engine is None:
            from rag_engine import RAGEngine

            rag_engine = RAGEngine()
        self.rag_engine = rag_engine
        self.batch_size = int(batch_size) if batch_size else 20
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.inter_request_delay_seconds = max(0.0, float(inter_request_delay_seconds))
        self.llm_max_retries = max(0, int(llm_max_retries))
        self.llm_base_backoff_seconds = max(0.1, float(llm_base_backoff_seconds))
        self.llm_max_backoff_seconds = max(1.0, float(llm_max_backoff_seconds))
        self._last_llm_call_ts: Optional[float] = None

    def process_all_vehicle_summary(
        self,
        query: str,
        vehicleid_collection: str,
        session_id: str = "default",
        claims: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del vehicleid_collection  # Interface compatibility; not required for all-vehicle summary flow.
        claims = claims or {}
        session_id = session_id or "default"
        query = str(query or "").strip()

        classification_label = self._classify_query(query=query, session_id=session_id)
        base_response: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "classification_label": classification_label,
            "processed": False,
            "skip_reason": None,
            "summary_collection": None,
            "total_records": 0,
            "total_unique_vehicles": 0,
            "batch_size": self.batch_size,
            "total_batches": 0,
            "partial_summaries": [],
            "final_summary": "",
            "errors": [],
        }

        if classification_label != "vehiclesummary-all":
            base_response["skip_reason"] = "classification_not_vehiclesummary_all"
            return base_response

        summary_collection = self._resolve_summary_collection(query)
        records = self._fetch_all_records(summary_collection="vehiclesummary_feb_collection_v3", claims=claims)
        grouped = self._group_records_by_vehicle(records)
        batches = self._chunk_vehicle_groups(grouped=grouped, batch_size=self.batch_size)

        base_response["processed"] = True
        base_response["summary_collection"] = summary_collection
        base_response["total_records"] = len(records)
        base_response["total_unique_vehicles"] = len(grouped)
        base_response["total_batches"] = len(batches)

        if not records:
            base_response["final_summary"] = "No relevant records were found in the available data."
            return base_response

        partial_summaries: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        total_batches = len(batches)
        for batch_index, batch_items in enumerate(batches, start=1):
            batch_context = self._build_batch_context(batch_items)
            if not batch_context:
                continue

            try:
                summary_text = self._summarize_batch(
                    batch_context=batch_context,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    query=query,
                )
                partial_summaries.append(
                    {
                        "batch_index": batch_index,
                        "vehicle_count": len(batch_items),
                        "vehicles": [vehicle_no for vehicle_no, _ in batch_items],
                        "summary": summary_text,
                    }
                )
            except Exception as exc:
                logger.error("Batch summarization failed for batch %s: %s", batch_index, exc)
                errors.append({"batch_index": batch_index, "error": str(exc)})

        if len(partial_summaries) == 1 and total_batches == 1:
            final_summary = str(partial_summaries[0].get("summary") or "").strip()
        elif partial_summaries:
            try:
                final_summary = self._summarize_partials(
                    partial_summaries=partial_summaries,
                    query=query,
                )
            except Exception as exc:
                logger.error("Final partial-summary synthesis failed: %s", exc)
                errors.append({"stage": "final_synthesis", "error": str(exc)})
                final_summary = "Unable to summarize due to processing errors."
        elif errors:
            final_summary = "Unable to summarize due to processing errors."
        else:
            final_summary = "No relevant records were found in the available data."

        base_response["partial_summaries"] = partial_summaries
        base_response["final_summary"] = final_summary
        base_response["errors"] = errors
        return base_response

    def _classify_query(self, query: str, session_id: str) -> str:
        query = str(query or "").strip()
        if not query:
            return "none"

        history_text = "No previous conversation."
        if hasattr(self.rag_engine, "_get_or_create_history") and hasattr(
            self.rag_engine, "_format_recent_history"
        ):
            try:
                history = self.rag_engine._get_or_create_history(session_id or "default")
                history_text = self.rag_engine._format_recent_history(history)
            except Exception:
                history_text = "No previous conversation."

        classifier_prompt = (
            "You are a strict query classifier.\n"
            "Classify the CURRENT QUERY into exactly one label from this list:\n"
            "generic, vehicledetail, vehiclesummary-all, vehiclesummary-notall, none\n\n"
            "Decision rules (apply in this order):\n"
            "1) If the user asks for summary/report/overview/status/health of vehicles:\n"
            "   - Use vehiclesummary-all for all vehicles/fleet/every vehicle.\n"
            "   - Use vehiclesummary-notall for one/some specific vehicles.\n"
            "2) If the user asks for vehicle identity/details and does NOT ask for summary/report/overview,\n"
            "   return vehicledetail.\n"
            "3) If the user asks a general vehicle-domain question not requesting vehicle lookup/summary,\n"
            "   return generic.\n"
            "4) If unrelated to vehicle domain, return none.\n\n"
            "Important:\n"
            "- Return only the label text, no explanation.\n\n"
            f"Chat history:\n{history_text}\n\n"
            f"Current query:\n{query}\n"
        )
        response = self._invoke_llm(classifier_prompt, stage_name="classification")
        answer_text = self._response_to_text(response)
        match = self._LABEL_PATTERN.search(answer_text)
        if not match:
            return "none"
        return match.group(1).lower()

    def _resolve_summary_collection(self, query: str) -> str:
        normalized = str(query or "").lower()
        resolved_month = None
        for token, month_value in self._MONTH_ALIASES.items():
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                resolved_month = month_value
                break
        if resolved_month is None:
            resolved_month = datetime.now().strftime("%b").lower()
        return f"vehiclesummary_{resolved_month}_collection_v3"

    def _fetch_all_records(
        self,
        summary_collection: str,
        claims: Optional[Dict[str, Any]] = None,
    ) -> List[NormalizedSummaryRecord]:
        claims = claims or {}
        claims_where = self.rag_engine._build_claims_where(claims)
        summary_db = self.rag_engine.get_or_create_collection(summary_collection)
        raw_records = summary_db.get(where=claims_where) if claims_where else summary_db.get()

        docs = raw_records.get("documents") or []
        metas = raw_records.get("metadatas") or []
        result_count = min(len(docs), len(metas)) if metas else len(docs)

        normalized_records: List[NormalizedSummaryRecord] = []
        for idx in range(result_count):
            document = str(docs[idx] or "").strip()
            if not document:
                continue

            metadata = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            vehicle_no = self._extract_vehicle_no(metadata)
            if vehicle_no == "UNKNOWN":
                vehicle_no = self._extract_vehicle_no_from_text(document)
            normalized_records.append(
                {"document": document, "metadata": metadata, "vehicle_no": vehicle_no}
            )
        return normalized_records

    @staticmethod
    def _extract_vehicle_no(metadata: Dict[str, Any]) -> str:
        for key in ("VEHICLE_NO", "vehicleNo", "vehicleno"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return "UNKNOWN"

    @staticmethod
    def _extract_vehicle_no_from_text(document: str) -> str:
        text = str(document or "")
        match = re.search(r"\bVehicle\s+([A-Za-z0-9_-]+)\b", text, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()
        return "UNKNOWN"

    # def _group_records_by_vehicle(
    #     self,
    #     records: List[NormalizedSummaryRecord],
    # ) -> Dict[str, List[str]]:
    #     grouped_docs: Dict[str, List[str]] = {}
    #     seen_docs_per_vehicle: Dict[str, set] = {}

    #     for record in records:
    #         vehicle_no = str(record.get("vehicle_no") or "UNKNOWN").strip() or "UNKNOWN"
    #         document = str(record.get("document") or "").strip()
    #         if not document:
    #             continue

    #         if vehicle_no not in grouped_docs:
    #             grouped_docs[vehicle_no] = []
    #             seen_docs_per_vehicle[vehicle_no] = set()

    #         if document in seen_docs_per_vehicle[vehicle_no]:
    #             continue
    #         seen_docs_per_vehicle[vehicle_no].add(document)
    #         grouped_docs[vehicle_no].append(document)
    #     return grouped_docs
    def _group_records_by_vehicle(
        self,
        records: List[NormalizedSummaryRecord],
    ) -> Dict[str, List[str]]:
        grouped_docs: Dict[str, List[str]] = {}
        seen_docs_per_vehicle: Dict[str, set] = {}

        for record in records:
            vehicle_id = str(record.get("vehicle_no") or "UNKNOWN").strip() or "UNKNOWN"
            document = str(record.get("document") or "").strip()

            if not document:
                continue

            if vehicle_id not in grouped_docs:
                grouped_docs[vehicle_id] = []
                seen_docs_per_vehicle[vehicle_id] = set()

            if document in seen_docs_per_vehicle[vehicle_id]:
                continue

            seen_docs_per_vehicle[vehicle_id].add(document)
            grouped_docs[vehicle_id].append(document)

        return grouped_docs

    @staticmethod
    def _group_list_input(grouped_list: List[Any]) -> Dict[str, List[str]]:
        grouped_docs: Dict[str, List[str]] = {}
        seen_docs: Dict[str, set] = {}
        vehicle_pattern = re.compile(r"\bVehicle\s+([A-Za-z0-9_-]+)\b", re.IGNORECASE)

        for item in grouped_list:
            text = str(item or "").strip()
            if not text:
                continue

            match = vehicle_pattern.search(text)
            vehicle_no = match.group(1).strip() if match else "UNKNOWN"
            if vehicle_no not in grouped_docs:
                grouped_docs[vehicle_no] = []
                seen_docs[vehicle_no] = set()

            if text in seen_docs[vehicle_no]:
                continue
            seen_docs[vehicle_no].add(text)
            grouped_docs[vehicle_no].append(text)
        return grouped_docs

    @staticmethod
    def _chunk_vehicle_groups(
        grouped: Any,
        batch_size: int = 20,
    ) -> List[List[Tuple[str, List[str]]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if isinstance(grouped, list):
            grouped = VehicleSummaryAllBatchProcessor._group_list_input(grouped)
        elif not isinstance(grouped, dict):
            grouped = {}

        ordered_items = sorted(grouped.items(), key=lambda item: item[0])
        batches: List[List[Tuple[str, List[str]]]] = []
        for start in range(0, len(ordered_items) , batch_size):
            batches.append(ordered_items[start : start + batch_size])
        return batches
    @staticmethod
    def _build_batch_context(batch_items: List[Tuple[str, List[str]]]) -> str:
        parts: List[str] = []
        for vehicle_no, docs in batch_items:
            merged_docs = "\n".join(docs)
            parts.append(f"VehicleNo: {vehicle_no}\nDocument:\n{merged_docs}")
        return "\n\n---\n\n".join(parts)

    def _summarize_batch(
        self,
        batch_context: str,
        batch_index: int,
        total_batches: int,
        query: str,
    ) -> str:
        prompt = (
            "You are a helpful assistant for vehicle analytics.\n"
            "Use ONLY the provided context.\n"
            "Create concise vehicle-wise summaries for this batch.\n"
            "Return one section per vehicle and avoid repeating vehicles.\n\n"
            f"Batch index: {batch_index}\n"
            f"Total batches: {total_batches}\n"
            f"Original user query: {query}\n\n"
            f"Context:\n{batch_context}\n\n"
            "Output format:\n"
            "**Vehicle: <VehicleNo>**\n"
            "1) Summary: <short summary>\n"
            "2) Summary points: <key values in bullets>\n"
        )
        response = self._invoke_llm(prompt, stage_name=f"batch_{batch_index}")
        time.sleep(5)  # throttle requests

        return self._response_to_text(response)

    def _summarize_partials(
        self,
        partial_summaries: List[Dict[str, Any]],
        query: str,
    ) -> str:
        partial_text_blocks: List[str] = []
        for item in partial_summaries:
            batch_index = item.get("batch_index")
            summary = str(item.get("summary") or "").strip()
            if summary:
                partial_text_blocks.append(f"Batch {batch_index}:\n{summary}")

        prompt = (
            "You are a helpful assistant for fleet-level consolidation.\n"
            "Use only the partial summaries from batches.\n"
            "Provide a final consolidated output with:\n"
            "1) Overall highlights\n"
            "2) Repeated patterns\n"
            "3) Notable vehicle-specific points\n\n"
            f"Original user query: {query}\n\n"
            "Partial summaries from all batches:\n"
            f"{chr(10).join(partial_text_blocks)}\n"
        )
        response = self._invoke_llm(prompt, stage_name="final_synthesis")
        return self._response_to_text(response)

    @staticmethod
    def _is_rate_limited_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        text = str(exc).lower()
        return "429" in text or "too many requests" in text or "rate limit" in text

    def _respect_inter_request_delay(self) -> None:
        if self.inter_request_delay_seconds <= 0:
            return
        if self._last_llm_call_ts is None:
            return
        elapsed = time.monotonic() - self._last_llm_call_ts
        sleep_for = self.inter_request_delay_seconds - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    def _invoke_llm(self, prompt: str, stage_name: str) -> Any:
        attempt = 0
        while True:
            try:
                self._respect_inter_request_delay()
                response = self.rag_engine.llm.invoke(prompt)
                self._last_llm_call_ts = time.monotonic()
                return response
            except Exception as exc:
                if not self._is_rate_limited_error(exc):
                    raise
                if attempt >= self.llm_max_retries:
                    raise
                backoff = min(
                    self.llm_max_backoff_seconds,
                    self.llm_base_backoff_seconds * (2 ** attempt),
                )
                logger.warning(
                    "Rate limited at stage '%s'. retry=%s/%s waiting %.1fs",
                    stage_name,
                    attempt + 1,
                    self.llm_max_retries,
                    backoff,
                )
                time.sleep(backoff)
                attempt += 1

    def _response_to_text(self, response: Any) -> str:
        if hasattr(self.rag_engine, "_llm_response_to_text"):
            return str(self.rag_engine._llm_response_to_text(response) or "").strip()
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(response, str):
            return response.strip()
        return str(response).strip()
