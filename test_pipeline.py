"""Quick test to verify the RAG pipeline works end-to-end."""
import sys
import os

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, os.path.dirname(__file__))

import config

print("[1/4] Config loaded OK")
print(f"  DATA_DIR:   {config.DATA_DIR}")
print(f"  RUNPOD API KEY: {'SET' if config.RUNPOD_API_KEY else 'NOT SET'}")
print(f"  MODEL:      {config.RUNPOD_MCQ_GENERATION_MODEL}")
print(f"  VECTOR STORE: {config.VECTOR_STORE_PROVIDER}")

# Test ingestion
print("\n[2/4] Testing ingestion...")
from rag.corpus import corpus_version
from rag.ingest import load_documents, split_documents, create_vector_store

docs = load_documents()
print(f"  Loaded {len(docs)} document(s)")
for d in docs:
    print(f"    - {d.metadata.get('source', 'unknown')}")

chunks = split_documents(docs, corpus_id=corpus_version())
print(f"  Split into {len(chunks)} chunk(s)")

create_vector_store(chunks)
print("  Vector store created!")

# Test retrieval
print("\n[3/4] Testing retrieval...")
from rag.retriever import retrieve_for_topic

results = retrieve_for_topic("Example Topic", "Remember", k=2)
print(f"  Retrieved {len(results)} chunk(s)")
for i, chunk in enumerate(results):
    print(f"  --- Chunk {i+1} ---")
    print(f"  {chunk[:150]}...")

# Test generation (just 1 question to save API calls)
print("\n[4/4] Testing MCQ generation...")
from rag.generator import generate_mcqs

questions = generate_mcqs(
    topic_name="Example Topic",
    difficulty_level="Remember",
    context_chunks=results,
    num_questions=2,
)
print(f"  Generated {len(questions)} question(s)")
for i, q in enumerate(questions):
    print(f"\n  Q{i+1}: {q.get('question', 'N/A')}")
    for opt in q.get('options', []):
        print(f"    {opt}")
    print(f"  Answer: {q.get('answer', 'N/A')}")
    print(f"  Explanation: {q.get('explanation', 'N/A')}")

print("\n=== ALL TESTS PASSED ===")
