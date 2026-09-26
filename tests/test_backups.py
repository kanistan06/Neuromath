import base64
import os

import config
from infrastructure.backups import _decrypt_file, _encrypt_file, _pg_dump_connection


def test_pg_dump_connection_keeps_password_out_of_process_arguments(monkeypatch):
    monkeypatch.setattr(
        config,
        "DATABASE_URL",
        "postgresql://student:p%40ssword@ep-test-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require",
    )

    arguments, environment = _pg_dump_connection()

    assert all("p@ssword" not in argument for argument in arguments)
    assert "--host=ep-test-pooler.us-east-2.aws.neon.tech" in arguments
    assert "--dbname=neondb" in arguments
    assert environment["PGPASSWORD"] == "p@ssword"
    assert environment["PGSSLMODE"] == "require"


def test_backup_files_are_authenticated_and_encrypted(monkeypatch, tmp_path):
    monkeypatch.setattr(
        config,
        "BACKUP_ENCRYPTION_KEY",
        base64.urlsafe_b64encode(os.urandom(32)).decode("ascii"),
    )
    source = tmp_path / "source.dump"
    encrypted = tmp_path / "source.dump.aes"
    restored = tmp_path / "restored.dump"
    source.write_bytes(b"private student database export")

    _encrypt_file(source, encrypted)
    _decrypt_file(encrypted, restored)

    assert source.read_bytes() not in encrypted.read_bytes()
    assert restored.read_bytes() == source.read_bytes()
