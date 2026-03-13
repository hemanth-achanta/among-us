from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from db.connection_manager import PrestoConnectionManager
from llm.llm_client import LLMClient
from config import config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "schema_definer" / "tables_config.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "schema.yaml"


@dataclass
class TableConfig:
    schema: str
    table: str
    name: str
    description: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TableConfig":
        schema = data.get("schema")
        table = data.get("table")
        if not schema or not table:
            raise ValueError("Each table entry must include 'schema' and 'table'.")
        name = data.get("name") or table
        description = data.get("description") or f"Auto-generated from hive.{schema}.{table}."
        return cls(schema=schema, table=table, name=name, description=description)


def load_tables_config(config_path: Path) -> list[TableConfig]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    tables_section = raw.get("tables")
    if not isinstance(tables_section, list) or not tables_section:
        raise ValueError("Config must contain a non-empty 'tables' list.")

    return [TableConfig.from_mapping(item) for item in tables_section]


def fetch_table_metadata(
    mgr: PrestoConnectionManager, cfg: TableConfig, sample_rows: int
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """
    Return (columns, samples) for hive.<schema>.<table>.

    columns: list of (name, type)
    samples: list of row dicts
    """
    fq_name = f"hive.{cfg.schema}.{cfg.table}"

    describe_sql = f"DESCRIBE {fq_name}"
    df_cols, _ = mgr.execute_query(describe_sql, max_rows=10_000)
    if df_cols.empty:
        raise RuntimeError(f"No column metadata returned for {fq_name}")

    if not {"Column", "Type"}.issubset(df_cols.columns):
        raise RuntimeError(
            f"Unexpected DESCRIBE result columns for {fq_name}: {list(df_cols.columns)}"
        )

    columns: list[tuple[str, str]] = []
    for _, row in df_cols.iterrows():
        col_name = str(row["Column"])
        col_type = str(row["Type"]).upper()
        if not col_name:
            continue
        columns.append((col_name, col_type))

    sample_sql = f"SELECT * FROM {fq_name} LIMIT {sample_rows}"
    df_samples, _ = mgr.execute_query(sample_sql, max_rows=sample_rows)
    samples = df_samples.to_dict(orient="records") if not df_samples.empty else []

    return columns, samples


def _generate_column_descriptions(
    llm: LLMClient,
    table_name: str,
    columns: list[tuple[str, str]],
    samples: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Use Claude (via LLMClient) to generate human-friendly descriptions
    for each column, based on its name, type, and a few sample rows.

    Output format is deliberately simple to avoid JSON parsing issues:
    one column per line, in the form:

        column_name | description text...
    """
    # Prepare compact text payload
    column_lines = "\n".join(f"- {name} ({ctype})" for name, ctype in columns)
    sample_preview = json.dumps(samples[:3], default=str)

    system_prompt = (
        "You are a data analyst helping document a Hive/Trino table.\n"
        "Given column names, data types, and a few sample rows, propose clear, "
        "concise, business-meaningful descriptions for each column.\n\n"
        "Return ONLY plain text, one column per line, in this exact format:\n"
        "column_name | description text here\n"
        "Do not include bullets, JSON, quotes, or any extra commentary."
    )

    user_message = (
        f"Table name: {table_name}\n\n"
        f"Columns:\n{column_lines}\n\n"
        f"Sample rows (JSON):\n{sample_preview}\n\n"
        "Now return the descriptions as specified."
    )

    try:
        response = llm.complete(
            model=config.LOW_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1024,
            temperature=0.1,
        )

        raw_text = response.content.strip()

        # Parse "name | description" lines
        result: dict[str, str] = {name: "" for name, _ in columns}
        for line in raw_text.splitlines():
            if "|" not in line:
                continue
            left, right = line.split("|", 1)
            col_name = left.strip()
            desc = right.strip()
            if col_name in result:
                result[col_name] = desc
        return result
    except Exception as exc:
        # Fallback: empty descriptions if anything goes wrong, but emit a hint.
        print(
            f"[schema_definer] Failed to generate descriptions for table "
            f"'{table_name}': {exc}"
        )
        return {name: "" for name, _ in columns}


def build_schema_dict(
    tables: list[TableConfig],
    mgr: PrestoConnectionManager,
    sample_rows: int,
    llm: LLMClient,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"tables": {}, "relationships": []}

    for cfg in tables:
        columns, samples = fetch_table_metadata(mgr, cfg, sample_rows)

        # Use fully-qualified physical table name as the schema key so the LLM
        # generates SQL with the correct catalog + schema automatically.
        table_key = f"{config.TRINO_CATALOG}.{cfg.schema}.{cfg.table}"

        # Ask Claude for column descriptions using samples as context
        descriptions = _generate_column_descriptions(llm, table_key, columns, samples)

        col_defs: dict[str, Any] = {}
        for col_name, col_type in columns:
            col_defs[col_name] = {
                "type": col_type,
                "description": descriptions.get(col_name, ""),
            }

        schema["tables"][table_key] = {
            "description": cfg.description,
            "columns": col_defs,
            "sample_rows": samples,
        }

    return schema


def write_schema_yaml(schema: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            schema,
            fh,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a schema YAML (schema.yaml) for configured "
            "Hive tables using the Presto/Trino connection."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to tables config YAML (default: schema_definer/tables_config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write generated schema YAML (default: schema.yaml in project root)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of sample rows to capture per table (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tables = load_tables_config(args.config)

    mgr = PrestoConnectionManager()
    llm = LLMClient()

    schema_dict = build_schema_dict(
        tables=tables,
        mgr=mgr,
        sample_rows=args.sample_rows,
        llm=llm,
    )

    write_schema_yaml(schema_dict, args.output)

    print(f"Schema written to: {args.output}")


if __name__ == "__main__":
    main()

