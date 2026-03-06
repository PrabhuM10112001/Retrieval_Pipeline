import unittest

from vehicle_detail_all_batched import VehicleDetailAllBatchProcessor


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
    def __init__(self, classifier_response="vehicledetail"):
        self.classifier_response = classifier_response

    def invoke(self, prompt):
        if "Classify the CURRENT QUERY" in prompt:
            return self.classifier_response
        return "none"


class FakeRAGEngine:
    def __init__(self, llm, vectordb_payload=None):
        self.llm = llm
        self.vectordb_payload = vectordb_payload or {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        self.last_collection_name = None

    def _get_or_create_history(self, _session_id):
        return FakeHistory()

    def _format_recent_history(self, _history):
        return "No previous conversation."

    def _llm_response_to_text(self, response):
        if isinstance(response, str):
            return response
        return str(response)

    def _build_claims_where_new(self, _claims):
        return None

    def _extract_vehicle_names_from_text(self, _query):
        return []

    def get_or_create_collection(self, collection_name):
        self.last_collection_name = collection_name
        return FakeVectorDB(self.vectordb_payload)


class VehicleDetailAllBatchProcessorTests(unittest.TestCase):
    def test_non_detail_query_is_skipped(self):
        rag = FakeRAGEngine(llm=FakeLLM(classifier_response="generic"))
        processor = VehicleDetailAllBatchProcessor(rag)

        result = processor.process_all_vehicle_detail(
            query="how to improve mileage",
            vehicle_detail_collection="vehicle_data_collection5",
        )

        self.assertFalse(result["processed"])
        self.assertEqual(result["skip_reason"], "classification_not_vehicledetail")

    def test_no_records_returns_not_found(self):
        rag = FakeRAGEngine(
            llm=FakeLLM(classifier_response="vehicledetail"),
            vectordb_payload={"ids": [], "documents": [], "metadatas": []},
        )
        processor = VehicleDetailAllBatchProcessor(rag)

        result = processor.process_all_vehicle_detail(
            query="show all vehicle details",
            vehicle_detail_collection="vehicle_data_collection5",
        )

        self.assertTrue(result["processed"])
        self.assertEqual(result["total_records"], 0)
        self.assertEqual(result["final_response"], "No relevant records were found in the available data.")

    def test_full_flow_returns_final_detail(self):
        rag = FakeRAGEngine(
            llm=FakeLLM(classifier_response="vehicledetail"),
            vectordb_payload={
                "ids": ["1", "2", "3"],
                "documents": ["d1", "d2", "d3"],
                "metadatas": [
                    {"VEHICLE_NO": "VH001", "VEHICLE_ID": "1", "MODEL": "M1"},
                    {"VEHICLE_NO": "VH002", "VEHICLE_ID": "2", "MODEL": "M2"},
                    {"VEHICLE_NO": "VH003", "VEHICLE_ID": "3", "MODEL": "M3"},
                ],
            },
        )
        processor = VehicleDetailAllBatchProcessor(rag, batch_size=2)

        result = processor.process_all_vehicle_detail(
            query="show all vehicle details",
            vehicle_detail_collection="vehicle_data_collection5",
        )

        self.assertTrue(result["processed"])
        self.assertEqual(result["total_unique_vehicles"], 3)
        self.assertEqual(result["total_batches"], 2)
        self.assertEqual(len(result["partial_details"]), 2)
        self.assertIn("Here is the consolidated list of vehicles without duplicates:", result["final_response"])
        self.assertIn("- VehicleNo: VH001, VehicleId: 1, Model: M1", result["final_response"])


if __name__ == "__main__":
    unittest.main()
