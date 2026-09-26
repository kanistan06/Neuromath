"""One-time migration of the existing NeuroMath SQLite application data to Neon.

This utility is intentionally separate from the web application's normal startup path.
It preserves primary keys so users, attempts, quiz history, settings, and related
records keep their existing relationships. The source SQLite database is read-only.

Usage from the project root:
    python scripts/migrate_sqlite_to_neon.py
    python scripts/migrate_sqlite_to_neon.py --execute

The target is taken from DATABASE_URL in the environment/.env and must be an empty
Neon PostgreSQL database. The application schema is created before copying data.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "instance" / "app.db"

# Parent tables always precede rows that reference them.
TABLE_ORDER = (
    "users",
    "auth_tokens",
    "user_settings",
    "attempts",
    "attempt_questions",
    "mistakes",
    "active_quizzes",
    "assessment_templates",
    "generation_drafts",
)


def _source_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _source_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _source_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def _load_environment() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - normal project install includes it.
        raise RuntimeError(
            "python-dotenv is required. Run `pip install -r requirements.txt` first."
        ) from exc

    load_dotenv(PROJECT_ROOT / ".env")
    database_url = str(os.getenv("DATABASE_URL", "")).strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required and must point to Neon PostgreSQL.")

    normalized = database_url.replace("postgresql+psycopg2://", "postgresql://")
    if normalized.startswith("postgres://"):
        normalized = "postgresql://" + normalized[len("postgres://") :]
    parsed = urlparse(normalized)
    if parsed.scheme != "postgresql" or not str(parsed.hostname or "").endswith(
        ".neon.tech"
    ):
        raise RuntimeError("DATABASE_URL must point to a Neon PostgreSQL database.")

    # Importing app creates the SQLAlchemy schema. The migration is an offline admin
    # operation, so production-only service guards must not prevent this one-time copy.
    if os.getenv("RAILWAY_ENVIRONMENT_NAME") or os.getenv("RAILWAY_PROJECT_ID"):
        raise RuntimeError("Run this migration from a trusted local/admin machine, not Railway.")
    os.environ["APP_ENV"] = "development"
    return database_url


def _convert_value(value: object, python_type: type | None) -> object:
    if value is None or python_type is None:
        return value
    if python_type is bool:
        return bool(value)
    if python_type is datetime and isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def _quoted_columns(columns: list[str]) -> str:
    return ", ".join(f'"{column}"' for column in columns)


def _named_parameters(columns: list[str]) -> str:
    return ", ".join(f":{column}" for column in columns)


def _target_is_empty(connection, text) -> tuple[bool, dict[str, int]]:
    counts: dict[str, int] = {}
    for table in TABLE_ORDER:
        count = int(connection.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar() or 0)
        counts[table] = count
    return not any(counts.values()), counts


def _reset_sequence(connection, text, table: str) -> None:
    maximum = connection.execute(text(f'SELECT MAX(id) FROM "{table}"')).scalar()
    if maximum is None:
        return
    sequence = connection.execute(
        text("SELECT pg_get_serial_sequence(:table_name, 'id')"),
        {"table_name": table},
    ).scalar()
    if sequence:
        connection.execute(
            text("SELECT setval(CAST(:sequence AS regclass), :value, true)"),
            {"sequence": str(sequence), "value": int(maximum)},
        )


def migrate(source_path: Path, *, execute: bool) -> int:
    if not source_path.exists() or not source_path.is_file():
        raise RuntimeError(f"SQLite source database was not found: {source_path}")

    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        available = _source_tables(source)
        missing = [table for table in TABLE_ORDER if table not in available]
        if missing:
            raise RuntimeError(
                "The SQLite database is missing expected NeuroMath tables: "
                + ", ".join(missing)
            )

        source_counts = {table: _source_count(source, table) for table in TABLE_ORDER}
        print(f"Source: {source_path}")
        for table in TABLE_ORDER:
            print(f"  {table}: {source_counts[table]} row(s)")

        _load_environment()
        if not execute:
            print(
                "\nPreflight only. No Neon data was changed. "
                "Run again with --execute after confirming DATABASE_URL."
            )
            return 0

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        # The import creates/updates the target schema using the existing application
        # model definitions. It does not change normal app behavior.
        import app as app_module
        from sqlalchemy import inspect, text

        # Flask-SQLAlchemy requires an active Flask application context before
        # db.engine, model metadata, or other context-bound database helpers
        # can be accessed from an offline script.
        with app_module.app.app_context():
            target_engine = app_module.db.engine
            if target_engine.dialect.name != "postgresql":
                raise RuntimeError("The target SQLAlchemy engine is not PostgreSQL.")

            inspector = inspect(target_engine)
            target_tables = set(inspector.get_table_names())
            missing_target = [table for table in TABLE_ORDER if table not in target_tables]
            if missing_target:
                raise RuntimeError(
                    "The Neon schema is missing expected tables: " + ", ".join(missing_target)
                )

            with target_engine.begin() as target:
                empty, target_counts = _target_is_empty(target, text)
                if not empty:
                    populated = ", ".join(
                        f"{table}={count}" for table, count in target_counts.items() if count
                    )
                    raise RuntimeError(
                        "Refusing to merge into a non-empty Neon database. "
                        f"Existing rows: {populated}. Use a new/empty Neon branch or database."
                    )

                migrated: dict[str, int] = {}
                for table in TABLE_ORDER:
                    source_columns = _source_columns(source, table)
                    target_column_info = {
                        str(column["name"]): column for column in inspector.get_columns(table)
                    }
                    columns = [
                        column for column in source_columns if column in target_column_info
                    ]
                    if not columns:
                        migrated[table] = 0
                        continue

                    rows = source.execute(
                        f'SELECT {_quoted_columns(columns)} FROM "{table}" ORDER BY id'
                    ).fetchall()
                    if rows:
                        records: list[dict[str, object]] = []
                        for row in rows:
                            record: dict[str, object] = {}
                            for column in columns:
                                column_type = target_column_info[column].get("type")
                                try:
                                    python_type = column_type.python_type
                                except (AttributeError, NotImplementedError):
                                    python_type = None
                                record[column] = _convert_value(row[column], python_type)
                            records.append(record)

                        statement = text(
                            f'INSERT INTO "{table}" ({_quoted_columns(columns)}) '
                            f'VALUES ({_named_parameters(columns)})'
                        )
                        target.execute(statement, records)
                    migrated[table] = len(rows)

                for table in TABLE_ORDER:
                    _reset_sequence(target, text, table)

            print("\nMigration completed successfully:")
            for table in TABLE_ORDER:
                print(f"  {table}: {migrated[table]} row(s)")
            print("The original SQLite database was not modified.")
            return 0
    finally:
        source.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely copy an existing NeuroMath SQLite database to an empty Neon database."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the existing SQLite app.db (default: instance/app.db).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the copy. Without this flag the script is preflight-only.",
    )
    args = parser.parse_args()
    try:
        return migrate(args.source.expanduser().resolve(), execute=args.execute)
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())