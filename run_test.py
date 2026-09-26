import traceback
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "result.txt")

def log(msg):
    with open(LOG, "a", encoding="ascii", errors="replace") as f:
        f.write(msg + "\n")

# Clear previous log
with open(LOG, "w", encoding="ascii") as f:
    f.write("")

try:
    log("STEP 1: Importing config")
    sys.path.insert(0, BASE)
    import config
    log(f"  RUNPOD API KEY set: {bool(config.RUNPOD_API_KEY)}")
    log(f"  Model: {config.RUNPOD_MCQ_GENERATION_MODEL}")

    log("STEP 2: Importing rag.ingest")
    from rag.corpus import corpus_version
    from rag.ingest import load_documents, split_documents, create_vector_store

    log("STEP 3: Loading documents")
    docs = load_documents()
    log(f"  Found {len(docs)} documents")

    log("STEP 4: Splitting documents")
    chunks = split_documents(docs, corpus_id=corpus_version())
    log(f"  Created {len(chunks)} chunks")

    log("STEP 5: Creating vector store")
    create_vector_store(chunks)
    log("  Vector store created!")

    log("STEP 6: Retrieving")
    from rag.retriever import retrieve_for_topic
    results = retrieve_for_topic("Example Topic", "Remember", k=2)
    log(f"  Retrieved {len(results)} chunks")
    for i, r in enumerate(results):
        safe_text = r[:120].encode("ascii", errors="replace").decode("ascii")
        log(f"  Chunk {i+1}: {safe_text}")

    log("STEP 7: Generating MCQs")
    from rag.generator import generate_mcqs
    questions = generate_mcqs(
        topic_name="Example Topic",
        difficulty_level="Remember",
        context_chunks=results,
        num_questions=2,
    )
    log(f"  Generated {len(questions)} questions")
    for i, q in enumerate(questions):
        safe_q = str(q.get("question", "N/A")).encode("ascii", errors="replace").decode("ascii")
        log(f"  Q{i+1}: {safe_q}")
        safe_a = str(q.get("answer", "N/A")).encode("ascii", errors="replace").decode("ascii")
        log(f"  Answer: {safe_a}")

    log("ALL STEPS PASSED")
except Exception:
    log("ERROR:")
    log(traceback.format_exc())
