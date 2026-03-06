import unittest
from datetime import datetime

from vehicle_summary_all_batched import VehicleSummaryAllBatchProcessor


class FakeHistory:
    messages = []


class FakeVectorDB:
    def __init__(self, payload):
        self.payload = payload
        self.last_where = None

    def get(self, where=None):
        self.last_where = where
        return self.payload


class FakeLLM:
    def __init__(self, responses=None, classifier_response="vehiclesummary-all"):
        self.responses = responses or {}
        self.classifier_response = classifier_response
        self.batch_counter = 0

    def invoke(self, prompt):
        if "Classify the CURRENT QUERY" in prompt:
            return self.classifier_response
        if "Batch index:" in prompt:
            self.batch_counter += 1
            if self.responses.get("raise_batch_1") and "Batch index: 1" in prompt:
                raise RuntimeError("Batch-1 failed")
            return f"batch-summary-{self.batch_counter}"
        if "Partial summaries from all batches" in prompt:
            if self.responses.get("raise_final"):
                raise RuntimeError("Final synthesis failed")
            return "final-summary"
        return "none"


class FakeRAGEngine:
    def __init__(self, llm, vectordb_payload=None):
        self.llm = llm
        self.vectordb_payload = vectordb_payload or {
            "documents": [],
            "metadatas": [],
        }
        self.last_collection_name = None
        self.last_claims = None

    def _get_or_create_history(self, _session_id):
        return FakeHistory()

    def _format_recent_history(self, _history):
        return "No previous conversation."

    def _llm_response_to_text(self, response):
        if isinstance(response, str):
            return response
        return str(response)

    def _build_claims_where(self, claims):
        self.last_claims = claims
        if not claims:
            return None
        return {"$or": [{"RESELLER_ID": claims.get("resellerId")}]}

    def get_or_create_collection(self, collection_name):
        self.last_collection_name = collection_name
        return FakeVectorDB(self.vectordb_payload)


class VehicleSummaryAllBatchProcessorTests(unittest.TestCase):
    def test_classifier_parsing_clean_label(self):
        rag = FakeRAGEngine(llm=FakeLLM(classifier_response="vehiclesummary-all"))
        processor = VehicleSummaryAllBatchProcessor(rag)
        self.assertEqual(
            processor._classify_query("summary for all vehicles", "s1"),
            "vehiclesummary-all",
        )

    def test_classifier_parsing_noisy_label(self):
        rag = FakeRAGEngine(llm=FakeLLM(classifier_response="Label: vehiclesummary-notall."))
        processor = VehicleSummaryAllBatchProcessor(rag)
        self.assertEqual(
            processor._classify_query("summary for this vehicle", "s1"),
            "vehiclesummary-notall",
        )

    def test_classifier_invalid_fallback_to_none(self):
        rag = FakeRAGEngine(llm=FakeLLM(classifier_response="unknown-category"))
        processor = VehicleSummaryAllBatchProcessor(rag)
        self.assertEqual(processor._classify_query("hello", "s1"), "none")

    def test_collection_resolution_with_explicit_month(self):
        rag = FakeRAGEngine(llm=FakeLLM())
        processor = VehicleSummaryAllBatchProcessor(rag)
        self.assertEqual(
            processor._resolve_summary_collection("show all vehicle summary for february"),
            "vehiclesummary_feb_collection_v3",
        )

    def test_collection_resolution_without_month_uses_current(self):
        rag = FakeRAGEngine(llm=FakeLLM())
        processor = VehicleSummaryAllBatchProcessor(rag)
        expected_month = datetime.now().strftime("%b").lower()
        self.assertEqual(
            processor._resolve_summary_collection("show all vehicle summary"),
            f"vehiclesummary_{expected_month}_collection_v3",
        )

    def test_grouping_and_batching_dedup_and_size(self):
        rag = FakeRAGEngine(llm=FakeLLM())
        processor = VehicleSummaryAllBatchProcessor(rag, batch_size=20)
        records = []
        for i in range(41):
            vehicle = f"VH{i:02d}"
            records.append({"document": "doc-a", "metadata": {"VEHICLE_NO": vehicle}, "vehicle_no": vehicle})
            records.append({"document": "doc-a", "metadata": {"vehicleNo": vehicle}, "vehicle_no": vehicle})
            records.append({"document": "doc-b", "metadata": {"vehicleno": vehicle}, "vehicle_no": vehicle})

        grouped = processor._group_records_by_vehicle(records)
        self.assertEqual(len(grouped), 41)
        self.assertEqual(grouped["VH00"], ["doc-a", "doc-b"])

        batches = processor._chunk_vehicle_groups(grouped, batch_size=20)
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[0]), 20)
        self.assertEqual(len(batches[1]), 20)
        self.assertEqual(len(batches[2]), 1)

    def test_chunk_vehicle_groups_accepts_list_input(self):
        rag = FakeRAGEngine(llm=FakeLLM())
        processor = VehicleSummaryAllBatchProcessor(rag, batch_size=2)
        grouped_list = [
            "Vehicle 43784 on 2026-02-01 moved for 10.0 minutes.",
            "Vehicle 43740 on 2026-02-01 moved for 20.0 minutes.",
            "Vehicle 43737 on 2026-02-01 moved for 30.0 minutes.",
            "Vehicle 43784 on 2026-02-01 moved for 10.0 minutes.",
        ]
        batches = processor._chunk_vehicle_groups(grouped=grouped_list, batch_size=2)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 1)

    def test_non_all_query_returns_skipped_status(self):
        llm = FakeLLM(classifier_response="generic")
        rag = FakeRAGEngine(llm=llm)
        processor = VehicleSummaryAllBatchProcessor(rag)
        result = processor.process_all_vehicle_summary(
            query="how to improve mileage",
            vehicleid_collection="vehicle_data_collection5",
            session_id="s1",
            claims=None,
        )
        self.assertFalse(result["processed"])
        self.assertEqual(result["skip_reason"], "classification_not_vehiclesummary_all")
        self.assertEqual(result["classification_label"], "generic")

    def test_no_records_returns_not_found_message(self):
        llm = FakeLLM(classifier_response="vehiclesummary-all")
        rag = FakeRAGEngine(
            llm=llm,
            vectordb_payload={"documents": [], "metadatas": []},
        )
        processor = VehicleSummaryAllBatchProcessor(rag)
        result = processor.process_all_vehicle_summary(
            query="show all vehicle summary",
            vehicleid_collection="vehicle_data_collection5",
            claims={},
        )
        self.assertTrue(result["processed"])
        self.assertEqual(result["total_records"], 0)
        self.assertEqual(result["partial_summaries"], [])
        self.assertEqual(result["final_summary"], "No relevant records were found in the available data.")

    def test_integration_full_flow(self):
        llm = FakeLLM(classifier_response="vehiclesummary-all")
        rag = FakeRAGEngine(
            llm=llm,
            vectordb_payload={
                "documents": ["d1", "d2", "d3"],
                "metadatas": [
                    {"VEHICLE_NO": "VH001"},
                    {"vehicleNo": "VH002"},
                    {"vehicleno": "VH003"},
                ],
            },
        )
        processor = VehicleSummaryAllBatchProcessor(rag, batch_size=2)
        result = processor.process_all_vehicle_summary(
            query="give all vehicle summary",
            vehicleid_collection="vehicle_data_collection5",
            claims={"resellerId": 100},
        )
        self.assertTrue(result["processed"])
        self.assertEqual(result["classification_label"], "vehiclesummary-all")
        self.assertEqual(result["total_unique_vehicles"], 3)
        self.assertEqual(result["total_batches"], 2)
        self.assertEqual(len(result["partial_summaries"]), 2)
        self.assertEqual(result["final_summary"], "final-summary")
        self.assertEqual(result["errors"], [])

    def test_integration_partial_failure_still_finalizes(self):
        llm = FakeLLM(responses={"raise_batch_1": True}, classifier_response="vehiclesummary-all")
        rag = FakeRAGEngine(
            llm=llm,
            vectordb_payload={
                "documents": ["d1", "d2", "d3"],
                "metadatas": [
                    {"VEHICLE_NO": "VH001"},
                    {"VEHICLE_NO": "VH002"},
                    {"VEHICLE_NO": "VH003"},
                ],
            },
        )
        processor = VehicleSummaryAllBatchProcessor(rag, batch_size=2)
        result = processor.process_all_vehicle_summary(
            query="give all vehicle summary",
            vehicleid_collection="vehicle_data_collection5",
            claims={},
        )
        self.assertTrue(result["processed"])
        self.assertEqual(len(result["partial_summaries"]), 1)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["final_summary"], "final-summary")

    def test_all_batch_failures_returns_fallback(self):
        class AllBatchFailLLM(FakeLLM):
            def invoke(self, prompt):
                if "Classify the CURRENT QUERY" in prompt:
                    return "vehiclesummary-all"
                if "Batch index:" in prompt:
                    raise RuntimeError("batch-failed")
                if "Partial summaries from all batches" in prompt:
                    return "should-not-be-called"
                return "none"

        rag = FakeRAGEngine(
            llm=AllBatchFailLLM(),
            vectordb_payload={
                "documents": ["d1", "d2"],
                "metadatas": [{"VEHICLE_NO": "V1"}, {"VEHICLE_NO": "V2"}],
            },
        )
        processor = VehicleSummaryAllBatchProcessor(rag, batch_size=1)
        result = processor.process_all_vehicle_summary(
            query="all vehicles summary",
            vehicleid_collection="vehicle_data_collection5",
        )
        self.assertEqual(result["partial_summaries"], [])
        self.assertGreaterEqual(len(result["errors"]), 1)
        self.assertEqual(result["final_summary"], "Unable to summarize due to processing errors.")


if __name__ == "__main__":
    unittest.main()
