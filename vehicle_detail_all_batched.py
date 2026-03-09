import re
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from logger import logger

if TYPE_CHECKING:
    from rag_engine import RAGEngine


NormalizedDetailRecord = Dict[str, Any]


class VehicleDetailAllBatchProcessor:
    _LABEL_PATTERN = re.compile(
        r"\b(generic|vehicledetail|vehiclesummary-all|vehiclesummary-notall|none)\b",
        re.IGNORECASE,
    )

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

    def process_all_vehicle_detail(
        self,
        query: str,
        vehicle_detail_collection: str,
        session_id: str = "default",
        claims: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        claims = claims or {}
        session_id = session_id or "default"
        query = str(query or "").strip()

        # classification_label = self._classify_query(query=query, session_id=session_id)
        base_response: Dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            # "classification_label": classification_label,
            "processed": False,
            "skip_reason": None,
            "vehicle_detail_collection": vehicle_detail_collection,
            "total_records": 0,
            "total_unique_vehicles": 0,
            "batch_size": self.batch_size,
            "total_batches": 0,
            "partial_details": [],
            "final_response": "",
            "errors": [],
            "results": [],
        }

        # if classification_label != "vehicledetail":
        #     base_response["skip_reason"] = "classification_not_vehicledetail"
        #     return base_response

        records = self._fetch_all_records(
            vehicle_detail_collection=vehicle_detail_collection,
            claims=claims,
        )

        explicit_vehicle_names = self._extract_vehicle_names_from_query(query)
        if explicit_vehicle_names:
            records = [
                rec for rec in records if str(rec.get("vehicle_no") or "") in explicit_vehicle_names
            ]

        grouped = self._group_records_by_vehicle(records)
        batches = self._chunk_vehicle_groups(grouped=grouped, batch_size=self.batch_size)

        base_response["processed"] = True
        base_response["total_records"] = len(records)
        base_response["total_unique_vehicles"] = len(grouped)
        base_response["total_batches"] = len(batches)
         
        if not records:
            base_response["final_response"] = "No relevant records were found in the available data."
            return base_response

        partial_details: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        all_results: List[Dict[str, Any]] = []
        total_batches = len(batches)

        for batch_index, batch_items in enumerate(batches, start=1):
            batch_results = [payload for _, payload in batch_items]
            all_results.extend(batch_results)

            try:
                batch_context = self._build_detail_batch_context(batch_items)
                detail_text = self._generate_detail_batch_llm(
                    batch_context=batch_context,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    query=query,
                )
            except Exception as exc:
                logger.error("Batch detail generation failed for batch %s: %s", batch_index, exc)
                errors.append({"batch_index": batch_index, "error": str(exc)})
                detail_text = self._format_batch_rows(batch_items)

            partial_details.append(
                {
                    "batch_index": batch_index,
                    "vehicle_count": len(batch_items),
                    "vehicles": [vehicle_no for vehicle_no, _ in batch_items],
                    "detail": detail_text,
                }
            )

        if len(partial_details) == 1 and total_batches == 1:
            final_response = str(partial_details[0].get("detail") or "").strip()
        elif partial_details:
            try:
                final_response = self._synthesize_detail_partials_llm(
                    partial_details=partial_details,
                    query=query,
                )
            except Exception as exc:
                logger.error("Final detail synthesis failed: %s", exc)
                errors.append({"stage": "final_synthesis", "error": str(exc)})
                final_response = self._format_consolidated_rows(all_results)
        elif errors:
            final_response = "Unable to process vehicle details due to errors."
        else:
            final_response = "No relevant records were found in the available data."

        base_response["partial_details"] = partial_details
        base_response["final_response"] = final_response
        base_response["errors"] = errors
        base_response["results"] = all_results
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

    def _fetch_all_records(
        self,
        vehicle_detail_collection: str,
        claims: Optional[Dict[str, Any]] = None,
    ) -> List[NormalizedDetailRecord]:
        claims = claims or {}
        build_claims = getattr(self.rag_engine, "_build_claims_where_new", None)
        if callable(build_claims):
            claims_where = build_claims(claims)
        else:
            claims_where = self.rag_engine._build_claims_where(claims)

        detail_db = self.rag_engine.get_or_create_collection(vehicle_detail_collection)
        raw_records = detail_db.get(where=claims_where) if claims_where else detail_db.get()

        ids = raw_records.get("ids") or []
        docs = raw_records.get("documents") or []
        metas = raw_records.get("metadatas") or []
        result_count = min(len(docs), len(metas)) if metas else len(docs)

        normalized_records: List[NormalizedDetailRecord] = []
        for idx in range(result_count):
            document = str(docs[idx] or "").strip()
            if not document:
                continue

            metadata = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
            vehicle_no = self._extract_vehicle_no(metadata)
            if vehicle_no == "UNKNOWN":
                vehicle_no = self._extract_vehicle_no_from_text(document)

            normalized_records.append(
                {
                    "id": ids[idx] if idx < len(ids) else None,
                    "document": document,
                    "metadata": metadata,
                    "vehicle_no": vehicle_no,
                }
            )
        return normalized_records

    def _extract_vehicle_names_from_query(self, query: str) -> List[str]:
        extractor = getattr(self.rag_engine, "_extract_vehicle_names_from_text", None)
        if callable(extractor):
            names = extractor(query)
            return [str(name).strip() for name in (names or []) if str(name).strip()]
        return []

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

    @staticmethod
    def _group_records_by_vehicle(
        records: List[NormalizedDetailRecord],
    ) -> Dict[str, Dict[str, Any]]:
        grouped_docs: Dict[str, Dict[str, Any]] = {}
        seen_docs_per_vehicle: Dict[str, set] = {}

        for record in records:
            vehicle_no = str(record.get("vehicle_no") or "UNKNOWN").strip() or "UNKNOWN"
            document = str(record.get("document") or "").strip()
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            record_id = record.get("id")
            if not document:
                continue

            if vehicle_no not in grouped_docs:
                grouped_docs[vehicle_no] = {
                    "id": record_id,
                    "metadata": metadata,
                    "vehicleNo": vehicle_no,
                    "documents": [],
                }
                seen_docs_per_vehicle[vehicle_no] = set()

            if document in seen_docs_per_vehicle[vehicle_no]:
                continue
            seen_docs_per_vehicle[vehicle_no].add(document)
            grouped_docs[vehicle_no]["documents"].append(document)

        return grouped_docs

    @staticmethod
    def _chunk_vehicle_groups(
        grouped: Dict[str, Dict[str, Any]],
        batch_size: int = 20,
    ) -> List[List[Tuple[str, Dict[str, Any]]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if not isinstance(grouped, dict):
            grouped = {}

        ordered_items = sorted(grouped.items(), key=lambda item: item[0])
        batches: List[List[Tuple[str, Dict[str, Any]]]] = []
        for start in range(0, len(ordered_items), batch_size):
            batches.append(ordered_items[start : start + batch_size])
        return batches

    @staticmethod
    def _extract_first_match(text: str, pattern: str) -> Optional[str]:
        if not text:
            return None
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        return str(match.group(1)).strip()

    def _resolve_vehicle_row(self, payload: Dict[str, Any]) -> Tuple[str, str, str]:
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        vehicle_no = str(payload.get("vehicleNo") or self._extract_vehicle_no(metadata) or "UNKNOWN").strip()

        vehicle_id = metadata.get("VEHICLE_ID") or metadata.get("vehicleId") or metadata.get("vehicleid")
        model = metadata.get("MODEL") or metadata.get("model")

        if not vehicle_id or not model:
            docs = payload.get("documents") or []
            first_doc = str(docs[0]).strip() if docs else ""
            if not vehicle_id:
                vehicle_id = self._extract_first_match(first_doc, r"\bvehicle[_\s]?id\s*[:=-]\s*([A-Za-z0-9_-]+)")
            if not model:
                model = self._extract_first_match(first_doc, r"\bmodel\s*[:=-]\s*([A-Za-z0-9 ._-]+)")

        return (
            vehicle_no or "UNKNOWN",
            str(vehicle_id).strip() if vehicle_id not in (None, "") else "N/A",
            str(model).strip() if model not in (None, "") else "N/A",
        )

    def _format_batch_rows(self, batch_items: List[Tuple[str, Dict[str, Any]]]) -> str:
        rows: List[str] = []
        for _, payload in batch_items:
            vehicle_no, vehicle_id, model = self._resolve_vehicle_row(payload)
            rows.append(
                f"- VehicleNo: {vehicle_no}, VehicleId: {vehicle_id}, Model: {model}"
            )
        return "\n".join(rows)

    @staticmethod
    def _truncate_text(value: str, max_chars: int = 220) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3]}..."

    def _build_detail_batch_context(
        self,
        batch_items: List[Tuple[str, Dict[str, Any]]],
    ) -> str:
        parts: List[str] = []
        for _, payload in batch_items:
            vehicle_no, vehicle_id, model = self._resolve_vehicle_row(payload)
            documents = payload.get("documents") or []
            snippets: List[str] = []
            for idx, doc in enumerate(documents[:2], start=1):
                snippet = self._truncate_text(str(doc), max_chars=220)
                if snippet:
                    snippets.append(f"{idx}) {snippet}")
            supporting_text = "\n".join(snippets) if snippets else "N/A"

            parts.append(
                "\n".join(
                    [
                        f"VehicleNo: {vehicle_no}",
                        f"VehicleId: {vehicle_id}",
                        f"Model: {model}",
                        "Supporting snippets:",
                        supporting_text,
                    ]
                )
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_strict_row_lines(text: str) -> List[str]:
        rows: List[str] = []
        seen: set = set()
        pattern = re.compile(
            r"^\s*-\s*VehicleNo:\s*(.+?),\s*VehicleId:\s*(.+?),\s*Model:\s*(.+?)\s*$",
            re.IGNORECASE,
        )
        for line in str(text or "").splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            normalized_line = (
                f"- VehicleNo: {match.group(1).strip()}, "
                f"VehicleId: {match.group(2).strip()}, "
                f"Model: {match.group(3).strip()}"
            )
            signature = normalized_line.upper()
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(normalized_line)
        return rows

    def _generate_detail_batch_llm(
        self,
        batch_context: str,
        batch_index: int,
        total_batches: int,
        query: str,
    ) -> str:
        prompt = (
            "You are a strict vehicle detail formatter.\n"
            "Use ONLY the provided context.\n"
            "Return vehicle detail rows in this exact format only:\n"
            "- VehicleNo: <value>, VehicleId: <value>, Model: <value>\n\n"
            "Rules:\n"
            "- One row per unique vehicle.\n"
            "- Deduplicate duplicates in this batch.\n"
            "- If VehicleId/Model is missing, use N/A.\n"
            "- Do not add headings, notes, or extra text.\n\n"
            f"Batch index: {batch_index}\n"
            f"Total batches: {total_batches}\n"
            f"Original user query: {query}\n\n"
            f"Context:\n{batch_context}\n"
        )
        response = self._invoke_llm(prompt, stage_name=f"detail_batch_{batch_index}")
        response_text = self._response_to_text(response)
        rows = self._extract_strict_row_lines(response_text)
        return "\n".join(rows) if rows else response_text

    def _synthesize_detail_partials_llm(
        self,
        partial_details: List[Dict[str, Any]],
        query: str,
    ) -> str:
        partial_text_blocks: List[str] = []
        for item in partial_details:
            batch_index = item.get("batch_index")
            detail_text = str(item.get("detail") or "").strip()
            if detail_text:
                partial_text_blocks.append(f"Batch {batch_index}:\n{detail_text}")

        prompt = (
            "You are a strict vehicle detail consolidator.\n"
            "Use only the provided batch outputs.\n"
            "Return rows in this exact format only:\n"
            "- VehicleNo: <value>, VehicleId: <value>, Model: <value>\n\n"
            "Rules:\n"
            "- One row per unique VehicleNo.\n"
            "- Prefer non-N/A values when duplicates have different values.\n"
            "- Keep output concise with rows only; no headings or notes.\n\n"
            f"Original user query: {query}\n\n"
            "Batch outputs:\n"
            f"{chr(10).join(partial_text_blocks)}\n"
        )
        response = self._invoke_llm(prompt, stage_name="detail_final_synthesis")
        response_text = self._response_to_text(response)
        rows = self._extract_strict_row_lines(response_text)
        return "\n".join(rows) if rows else response_text

    def _format_consolidated_rows(self, all_results: List[Dict[str, Any]]) -> str:
        dedup: Dict[str, Dict[str, Any]] = {}
        for payload in all_results:
            vehicle_no, vehicle_id, model = self._resolve_vehicle_row(payload)
            key = vehicle_no.upper()
            if key not in dedup:
                dedup[key] = {
                    "VehicleNo": vehicle_no,
                    "VehicleId": vehicle_id,
                    "Model": model,
                }
                continue
            if dedup[key]["VehicleId"] == "N/A" and vehicle_id != "N/A":
                dedup[key]["VehicleId"] = vehicle_id
            if dedup[key]["Model"] == "N/A" and model != "N/A":
                dedup[key]["Model"] = model

        if not dedup:
            return "No relevant records were found in the available data."

        lines = ["Here is the consolidated list of vehicles without duplicates:", ""]
        for vehicle_no in sorted(dedup.keys()):
            row = dedup[vehicle_no]
            lines.append(
                f"- VehicleNo: {row['VehicleNo']}, VehicleId: {row['VehicleId']}, Model: {row['Model']}"
            )
        return "\n".join(lines)

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
