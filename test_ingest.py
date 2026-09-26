import traceback
try:
    from rag.ingest import ingest_all
    ingest_all()
    print("Ingestion success!")
except Exception as e:
    print("INGESTION ERROR:")
    traceback.print_exc()
