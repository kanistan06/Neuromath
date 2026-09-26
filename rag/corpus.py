"""Corpus identity and source manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import config


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(source_files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    project_root = config.PROJECT_ROOT.resolve()
    for path in sorted(source_files, key=lambda item: str(item).lower()):
        resolved = path.resolve()
        try:
            name = resolved.relative_to(project_root).as_posix()
        except ValueError:
            name = resolved.name
        rows.append(
            {
                "path": name,
                "size": resolved.stat().st_size,
                "sha256": file_digest(resolved),
            }
        )
    return rows


def corpus_version(manifest: list[dict[str, object]]) -> str:
    payload = {
        "label": config.CORPUS_LABEL,
        "sources": manifest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{config.CORPUS_LABEL}-{hashlib.sha256(encoded).hexdigest()[:16]}"
