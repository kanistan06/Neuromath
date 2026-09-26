"""Neon PostgreSQL backups to Cloudflare R2."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import boto3
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import config
from rag.ingest import read_ingest_manifest


_ENCRYPTED_FILE_HEADER = b"NEUROMATH-BACKUP-V1\n"


def _required_configuration() -> None:
    required = {
        "DATABASE_URL": config.DATABASE_URL,
        "R2_ENDPOINT_URL": config.R2_ENDPOINT_URL,
        "R2_ACCESS_KEY_ID": config.R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": config.R2_SECRET_ACCESS_KEY,
        "R2_BUCKET_NAME": config.R2_BUCKET_NAME,
        "BACKUP_ENCRYPTION_KEY": config.BACKUP_ENCRYPTION_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing backup configuration: {', '.join(missing)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encryption_key() -> bytes:
    try:
        value = config.BACKUP_ENCRYPTION_KEY
        padding = "=" * (-len(value) % 4)
        key = base64.urlsafe_b64decode(value + padding)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("BACKUP_ENCRYPTION_KEY must be valid base64.") from exc
    if len(key) != 32:
        raise RuntimeError("BACKUP_ENCRYPTION_KEY must decode to exactly 32 bytes.")
    return key


def _encrypt_file(source: Path, destination: Path) -> None:
    nonce = os.urandom(12)
    ciphertext = AESGCM(_encryption_key()).encrypt(
        nonce,
        source.read_bytes(),
        _ENCRYPTED_FILE_HEADER,
    )
    destination.write_bytes(_ENCRYPTED_FILE_HEADER + nonce + ciphertext)


def _decrypt_file(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    if not payload.startswith(_ENCRYPTED_FILE_HEADER):
        raise RuntimeError("The backup is not a supported NeuroMath encrypted export.")
    offset = len(_ENCRYPTED_FILE_HEADER)
    nonce = payload[offset : offset + 12]
    ciphertext = payload[offset + 12 :]
    if len(nonce) != 12 or not ciphertext:
        raise RuntimeError("The encrypted backup is incomplete.")
    plaintext = AESGCM(_encryption_key()).decrypt(
        nonce,
        ciphertext,
        _ENCRYPTED_FILE_HEADER,
    )
    destination.write_bytes(plaintext)


def _pg_dump_connection() -> tuple[list[str], dict[str, str]]:
    parsed = urlparse(config.DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://"))
    if parsed.scheme not in {"postgresql", "postgres"} or not parsed.hostname:
        raise RuntimeError("DATABASE_URL must be a PostgreSQL connection string.")
    query = parse_qs(parsed.query)
    arguments = [
        f"--host={parsed.hostname}",
        f"--port={parsed.port or 5432}",
        f"--username={unquote(parsed.username or '')}",
        f"--dbname={unquote(parsed.path.lstrip('/'))}",
    ]
    environment = {
        "PGPASSWORD": unquote(parsed.password or ""),
        "PGSSLMODE": query.get("sslmode", ["require"])[0],
    }
    return arguments, environment


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=config.R2_ENDPOINT_URL,
        aws_access_key_id=config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def _upload(client, path: Path, key: str, *, content_type: str) -> dict[str, object]:
    digest = _sha256(path)
    client.upload_file(
        str(path),
        config.R2_BUCKET_NAME,
        key,
        ExtraArgs={
            "ContentType": content_type,
            "Metadata": {"sha256": digest, "service": "neuromath"},
        },
    )
    return {"key": key, "size": path.stat().st_size, "sha256": digest}


def _remove_expired(client, now: datetime) -> int:
    cutoff = now - timedelta(days=config.BACKUP_RETENTION_DAYS)
    paginator = client.get_paginator("list_objects_v2")
    expired: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=config.R2_BUCKET_NAME, Prefix=f"{config.R2_PREFIX}/"):
        for item in page.get("Contents", []):
            if item.get("LastModified") and item["LastModified"] < cutoff:
                expired.append({"Key": item["Key"]})
    for start in range(0, len(expired), 1000):
        client.delete_objects(
            Bucket=config.R2_BUCKET_NAME,
            Delete={"Objects": expired[start : start + 1000], "Quiet": True},
        )
    return len(expired)


def _send_heartbeat() -> None:
    if not config.BETTER_STACK_HEARTBEAT_URL:
        return
    requests.get(config.BETTER_STACK_HEARTBEAT_URL, timeout=10).raise_for_status()


def run_backup() -> dict[str, object]:
    _required_configuration()
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{config.R2_PREFIX}/{now:%Y/%m/%d}/{timestamp}"
    client = _r2_client()

    with tempfile.TemporaryDirectory(prefix="neuromath-backup-") as directory:
        temporary = Path(directory)
        dump_file = temporary / f"neuromath-{timestamp}.dump"
        encrypted_dump = temporary / f"{dump_file.name}.aes"
        manifest_file = temporary / f"neuromath-{timestamp}.json"
        environment = os.environ.copy()
        connection_arguments, connection_environment = _pg_dump_connection()
        environment.update(connection_environment)
        subprocess.run(
            [
                config.PG_DUMP_PATH,
                *connection_arguments,
                "--format=custom",
                "--compress=9",
                "--no-owner",
                "--no-acl",
                f"--file={dump_file}",
            ],
            env=environment,
            check=True,
            timeout=1800,
        )
        _encrypt_file(dump_file, encrypted_dump)
        uploaded = [
            _upload(
                client,
                encrypted_dump,
                f"{prefix}/{encrypted_dump.name}",
                content_type="application/octet-stream",
            )
        ]
        paper_file = config.PROJECT_ROOT / "generated_paper.json"
        if paper_file.is_file():
            encrypted_paper = temporary / "generated_paper.json.aes"
            _encrypt_file(paper_file, encrypted_paper)
            uploaded.append(
                _upload(
                    client,
                    encrypted_paper,
                    f"{prefix}/{encrypted_paper.name}",
                    content_type="application/octet-stream",
                )
            )
        ingest = read_ingest_manifest()
        manifest = {
            "created_at": now.isoformat(),
            "service": "NeuroMath",
            "database": "Neon PostgreSQL",
            "storage": "Cloudflare R2",
            "encryption": "AES-256-GCM",
            "corpus_version": ingest.get("corpus_version"),
            "chunker_version": ingest.get("chunker_version"),
            "files": uploaded,
        }
        manifest_file.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        uploaded_manifest = _upload(
            client,
            manifest_file,
            f"{prefix}/{manifest_file.name}",
            content_type="application/json",
        )

    removed = _remove_expired(client, now)
    _send_heartbeat()
    return {
        "status": "ok",
        "backup_prefix": prefix,
        "manifest": uploaded_manifest,
        "expired_objects_removed": removed,
    }


def restore_backup(object_key: str, *, confirm_database: str) -> dict[str, object]:
    _required_configuration()
    connection_arguments, connection_environment = _pg_dump_connection()
    database_name = next(
        argument.removeprefix("--dbname=")
        for argument in connection_arguments
        if argument.startswith("--dbname=")
    )
    if confirm_database != database_name:
        raise RuntimeError(
            "Restore confirmation must exactly match the destination database name."
        )
    if not object_key.startswith(f"{config.R2_PREFIX}/") or not object_key.endswith(".dump.aes"):
        raise RuntimeError("Restore key must identify an encrypted dump inside R2_PREFIX.")

    with tempfile.TemporaryDirectory(prefix="neuromath-restore-") as directory:
        temporary = Path(directory)
        encrypted_dump = temporary / "backup.dump.aes"
        dump_file = temporary / "backup.dump"
        _r2_client().download_file(
            config.R2_BUCKET_NAME,
            object_key,
            str(encrypted_dump),
        )
        _decrypt_file(encrypted_dump, dump_file)
        environment = os.environ.copy()
        environment.update(connection_environment)
        subprocess.run(
            [
                config.PG_RESTORE_PATH,
                *connection_arguments,
                "--clean",
                "--if-exists",
                "--no-owner",
                "--no-acl",
                str(dump_file),
            ],
            env=environment,
            check=True,
            timeout=1800,
        )
    return {"status": "ok", "restored_key": object_key, "database": database_name}
