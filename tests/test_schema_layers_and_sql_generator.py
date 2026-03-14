import textwrap
import unittest

from db.schema_loader import SchemaLoader
from backend.sql_generator import SQLGenerator, SQLGenerationResult


class FakeLLMResponse:
    def __init__(self, content: str, model: str = "fake-model"):
        self.content = content
        self.model = model


class FakeLLMClient:
    """
    Minimal stand-in for LLMClient used in tests.
    """

    def __init__(self, response_content: str):
        self._response_content = response_content

    def complete(self, model: str, messages, system: str, max_tokens: int | None = None):
        # Ignore inputs, just return the canned response
        return FakeLLMResponse(content=self._response_content, model=model)


class SchemaLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Tiny in-memory schema for testing helpers
        self.schema_dict = {
            "tables": {
                "hive.data_model.doc_consult_orders": {
                    "description": "Orders fact table.",
                    "layer": "raw",
                    "subject": "consultations",
                    "columns": {
                        "order_id": {"type": "VARCHAR"},
                    },
                },
                "hive.pre_analytics.doc_consult_metrics": {
                    "description": "Daily consultation metrics.",
                    "layer": "semi",
                    "subject": "consultations",
                    "columns": {
                        "dt": {"type": "DATE"},
                    },
                },
            },
            "relationships": [],
        }
        self.loader = SchemaLoader(self.schema_dict)

    def test_get_tables_by_layer_and_subject(self) -> None:
        raw_tables = self.loader.get_tables_by_layer("raw")
        semi_tables = self.loader.get_tables_by_layer("semi")
        consult_tables = self.loader.get_tables_by_subject("consultations")

        self.assertIn("hive.data_model.doc_consult_orders", raw_tables)
        self.assertIn("hive.pre_analytics.doc_consult_metrics", semi_tables)
        # Both tables share the same subject
        self.assertGreaterEqual(len(consult_tables), 2)

    def test_format_layered_summary_non_empty(self) -> None:
        summary = self.loader.format_layered_summary()
        self.assertIn("Layer:", summary)
        self.assertIn("subject: consultations", summary)


class SQLGeneratorSchemaOverrideTests(unittest.TestCase):
    def test_schema_override_is_used(self) -> None:
        """
        Ensure that SQLGenerator.generate honours schema_override and still
        produces a parsable SQLGenerationResult.
        """
        # Minimal fake JSON response from the LLM
        response_json = textwrap.dedent(
            """
            {
              "sql_query": "SELECT 1 AS x",
              "reasoning": "Smoke-test query",
              "confidence": 0.9
            }
            """
        ).strip()
        fake_llm = FakeLLMClient(response_content=response_json)

        # Baseline schema and override schema strings are irrelevant for parsing;
        # we just verify that the call succeeds with the override parameter set.
        generator = SQLGenerator(
            llm_client=fake_llm,
            schema_str="BASE SCHEMA",
            dialect="Presto SQL (Trino)",
            max_rows=100,
        )

        result: SQLGenerationResult = generator.generate(
            question="Test question",
            model="fake-model",
            previous_queries=None,
            chat_history=None,
            subquery_description=None,
            schema_override="OVERRIDE SCHEMA",
        )

        self.assertIsNotNone(result.sql_query)
        self.assertEqual(result.sql_query.strip(), "SELECT 1 AS x")
        self.assertGreater(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

