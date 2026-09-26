"""
Main entry point for the RAG MCQ Assessment System.

Ties together: ingest → generate paper → record answers → recommend topics.
"""

import json
import sys

import config


def cmd_ingest(force: bool = False):
    """Ingest study material into the vector store."""
    from rag.ingest import ingest_all
    result = ingest_all(force=force)
    print(json.dumps(result, indent=2))


def cmd_import_question_bank(arguments):
    import argparse
    from rag.question_bank_import import import_question_bank

    parser = argparse.ArgumentParser(prog="python main.py import-question-bank")
    parser.add_argument("--source", required=True)
    parser.add_argument("--report", default="reports/question-bank-import.json")
    parsed = parser.parse_args(arguments)
    result = import_question_bank(parsed.source, report_path=parsed.report)
    print(json.dumps({key: value for key, value in result.items() if key != "source_audit"}, indent=2))


def cmd_check_embeddings():
    import math
    from rag.embeddings import build_embeddings

    vectors = build_embeddings().embed_documents(["Find the perimeter of a rectangle.", "Solve a quadratic equation."])
    if len(vectors) != 2 or any(len(vector) != config.HF_EMBEDDING_DIMENSIONS for vector in vectors):
        raise RuntimeError("Embedding smoke check returned an unexpected shape.")
    print(json.dumps({"ready": True, "provider": config.HF_EMBEDDING_PROVIDER,
                      "model": config.HF_EMBEDDING_MODEL, "dimensions": len(vectors[0]),
                      "norms": [round(math.hypot(*vector), 6) for vector in vectors]}, indent=2))


def cmd_check_index():
    """Validate Grade 10/11 vector coverage without generating a paper."""
    from rag.ingest import vector_index_status

    status = vector_index_status()
    print(json.dumps(status, indent=2))
    if not status.get("ready"):
        raise SystemExit(1)


def cmd_validate_question_bank():
    """Validate past-paper records and their reviewed curriculum mappings."""
    from rag.question_bank import validate_question_bank

    status = validate_question_bank()
    print(json.dumps(status, indent=2))
    if not status.get("ready"):
        raise SystemExit(1)


def cmd_generate():
    """Generate an MCQ paper from the syllabus."""
    from rag.diagnostic import build_diagnostic_syllabus
    from rag.retriever import retrieve_for_syllabus
    from rag.generator import generate_paper
    from rag.adaptive import select_diagnostic_syllabus
    import secrets

    syllabus = select_diagnostic_syllabus(build_diagnostic_syllabus(), student_key="cli", variant=secrets.token_hex(16))

    print("🔍 Retrieving relevant content...")
    content = retrieve_for_syllabus(syllabus)

    print("📝 Generating MCQ paper...")
    paper = generate_paper(syllabus, content)

    output_path = config.PROJECT_ROOT / "generated_paper.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(paper, f, indent=2, ensure_ascii=False)

    print(f"✅ Paper saved to {output_path} ({len(paper)} questions)")


def cmd_build_question_pool(arguments):
    import argparse

    from app import app
    from rag.question_pool import build_question_pool

    parser = argparse.ArgumentParser(prog="python main.py build-question-pool")
    parser.add_argument(
        "--per-topic-level",
        type=int,
        default=config.QUESTION_POOL_TARGET_PER_TOPIC_LEVEL,
    )
    parser.add_argument("--grades", default="6,7,8,9,10,11")
    parser.add_argument("--levels", default="3,4,5")
    parser.add_argument("--topic", default="")
    parser.add_argument("--skip-past-papers", action="store_true")
    parsed = parser.parse_args(arguments)
    grades = [int(value.strip()) for value in parsed.grades.split(",") if value.strip()]
    levels = [int(value.strip()) for value in parsed.levels.split(",") if value.strip()]
    with app.app_context():
        result = build_question_pool(
            per_topic_level=max(1, parsed.per_topic_level),
            grades=grades,
            levels=levels,
            topic_filter=parsed.topic,
            include_past_papers=not parsed.skip_past_papers,
        )
    print(json.dumps(result, indent=2))


def cmd_question_pool_status():
    from app import app
    from rag.question_pool import pool_status

    with app.app_context():
        result = pool_status()
    print(json.dumps(result, indent=2))
    if not result.get("ready"):
        raise SystemExit(1)


def cmd_generate_template():
    """Generate, validate, and persist the production assessment template."""
    from app import (
        AssessmentTemplate,
        _artifact_versions,
        _diagnostic_fingerprint,
        _generate_and_save_paper,
        _student_diagnostic_syllabus,
        app,
    )

    with app.app_context():
        fingerprint = _diagnostic_fingerprint(_student_diagnostic_syllabus(None))
        payload = _generate_and_save_paper(None, force=True)
        versions = _artifact_versions()
        template = AssessmentTemplate.query.filter_by(
            fingerprint=fingerprint,
            **versions,
        ).first()
    print(
        json.dumps(
            {
                "ready": template is not None,
                "fingerprint": fingerprint,
                "question_count": payload["count"],
                **versions,
            },
            indent=2,
        )
    )


def cmd_backup():
    """Export Neon PostgreSQL and upload the backup to Cloudflare R2."""
    from infrastructure.backups import run_backup

    print(json.dumps(run_backup(), indent=2))


def cmd_restore_backup(object_key: str, confirm_database: str):
    """Restore one encrypted R2 database export after explicit confirmation."""
    from infrastructure.backups import restore_backup

    print(
        json.dumps(
            restore_backup(object_key, confirm_database=confirm_database),
            indent=2,
        )
    )


def cmd_record(student_id: str):
    """Record student answers and track mistakes."""
    from guidance.tracker import MistakeTracker

    paper_path = config.PROJECT_ROOT / "generated_paper.json"
    if not paper_path.exists():
        print("❌ No generated paper found. Run 'generate' first.")
        return

    with open(paper_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    # TODO: Replace with actual answer collection (CLI prompts, web form, etc.)
    print(f"📋 Paper has {len(paper)} questions.")
    print("   (Answer recording is a placeholder — implement your UI here.)")

    # Placeholder: simulate all correct for demo
    answers = {i: q.get("answer", "A") for i, q in enumerate(paper)}

    tracker = MistakeTracker(student_id)
    result = tracker.record_attempt(paper, answers)
    print(f"✅ Recorded: {result['correct']}/{result['total_questions']} correct "
          f"({result['score_percent']}%)")


def cmd_recommend(student_id: str):
    """Show study guidance based on past mistakes."""
    from guidance.tracker import MistakeTracker
    from guidance.analyzer import WeaknessAnalyzer
    from guidance.recommender import TopicRecommender

    tracker = MistakeTracker(student_id)
    analyzer = WeaknessAnalyzer(tracker)
    recommender = TopicRecommender(analyzer)
    recommender.print_guidance()


def main():
    """CLI dispatcher."""
    usage = """
RAG MCQ Assessment System
========================
Usage:
    python main.py ingest [--force]       — Ingest study material
    python main.py import-question-bank --source <directory> [--report <path>]
    python main.py check-embeddings       — Verify Hugging Face sentence embeddings
    python main.py check-index            — Validate Grade 10/11 vector coverage
    python main.py validate-question-bank — Validate past-paper records and mappings
    python main.py generate               — Generate an MCQ paper
    python main.py generate-template      — Generate and cache the production template
    python main.py build-question-pool [--per-topic-level 25] [--grades 6,7,8,9,10,11]
                                          — Pre-generate the global MCQ pool
    python main.py question-pool-status   — Show pre-generated pool coverage
    python main.py backup                 — Back up Neon PostgreSQL to Cloudflare R2
    python main.py restore-backup <key> --confirm-database <name>
                                          — Restore one encrypted R2 database export
    python main.py record <student_id>    — Record student answers
    python main.py recommend <student_id> — Show study guidance
    """

    if len(sys.argv) < 2:
        print(usage)
        return

    command = sys.argv[1].lower()

    if command == "ingest":
        cmd_ingest(force="--force" in sys.argv[2:])
    elif command == "import-question-bank":
        cmd_import_question_bank(sys.argv[2:])
    elif command == "check-embeddings":
        cmd_check_embeddings()
    elif command == "check-index":
        cmd_check_index()
    elif command == "validate-question-bank":
        cmd_validate_question_bank()
    elif command == "generate":
        cmd_generate()
    elif command == "generate-template":
        cmd_generate_template()
    elif command == "build-question-pool":
        cmd_build_question_pool(sys.argv[2:])
    elif command == "question-pool-status":
        cmd_question_pool_status()
    elif command == "backup":
        cmd_backup()
    elif command == "restore-backup":
        if len(sys.argv) < 5 or sys.argv[3] != "--confirm-database":
            raise SystemExit(
                "Usage: python main.py restore-backup <key> "
                "--confirm-database <name>"
            )
        cmd_restore_backup(sys.argv[2], sys.argv[4])
    elif command == "record":
        student_id = sys.argv[2] if len(sys.argv) > 2 else "default_student"
        cmd_record(student_id)
    elif command == "recommend":
        student_id = sys.argv[2] if len(sys.argv) > 2 else "default_student"
        cmd_recommend(student_id)
    else:
        print(f"❌ Unknown command: {command}")
        print(usage)


if __name__ == "__main__":
    main()
