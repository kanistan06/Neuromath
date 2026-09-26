import traceback
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
LOG = os.path.join(os.path.dirname(__file__), "debug.log")

with open(LOG, "w", encoding="utf-8") as f:
    try:
        f.write("Step 1: importing config...\n")
        f.flush()
        import config
        f.write(f"  DATA_DIR: {config.DATA_DIR}\n")
        f.write(f"  RUNPOD API KEY SET: {bool(config.RUNPOD_API_KEY)}\n")
        f.flush()

        f.write("Step 2: importing ingest...\n")
        f.flush()
        from rag.corpus import corpus_version
        from rag.ingest import load_documents, split_documents, create_vector_store

        f.write("Step 3: loading documents...\n")
        f.flush()
        docs = load_documents()
        f.write(f"  Loaded {len(docs)} doc(s)\n")
        f.flush()

        f.write("Step 4: splitting...\n")
        f.flush()
        chunks = split_documents(docs, corpus_id=corpus_version())
        f.write(f"  Created {len(chunks)} chunk(s)\n")
        f.flush()

        f.write("Step 5: creating vector store...\n")
        f.flush()
        create_vector_store(chunks)
        f.write("  Vector store created OK!\n")
        f.flush()

        f.write("Step 6: testing retrieval...\n")
        f.flush()
        from rag.retriever import retrieve_for_topic
        results = retrieve_for_topic("Example Topic", "Remember", k=2)
        f.write(f"  Retrieved {len(results)} chunk(s)\n")
        for i, r in enumerate(results):
            f.write(f"  Chunk {i+1}: {r[:100]}...\n")
        f.flush()

        f.write("Step 7: generating MCQs...\n")
        f.flush()
        from rag.generator import generate_mcqs
        questions = generate_mcqs(
            topic_name="Example Topic",
            difficulty_level="Remember",
            context_chunks=results,
            num_questions=2,
        )
        f.write(f"  Generated {len(questions)} question(s)\n")
        for i, q in enumerate(questions):
            f.write(f"  Q{i+1}: {q.get('question', 'N/A')}\n")
            for opt in q.get('options', []):
                f.write(f"    {opt}\n")
            f.write(f"  Answer: {q.get('answer', 'N/A')}\n")
        f.flush()

        f.write("\n=== ALL STEPS PASSED ===\n")
    except Exception as e:
        f.write(f"\n!!! ERROR !!!\n")
        f.write(traceback.format_exc())
        f.flush()
