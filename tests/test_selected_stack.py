import config


def test_selected_models_and_retrieval_dimensions_are_fixed():
    assert config.SELECTED_LLM_MODEL == "Qwen/Qwen2.5-Math-7B-Instruct"
    assert config.SELECTED_EMBEDDING_MODEL == "BAAI/bge-m3"
    assert config.SELECTED_EMBEDDING_DIMENSIONS == 1024
    assert config.RUNPOD_LLM_MODEL == config.SELECTED_LLM_MODEL
    assert config.HF_EMBEDDING_MODEL == config.SELECTED_EMBEDDING_MODEL
    assert hasattr(config, "RUNPOD_MCQ_ENDPOINT_ID")
    assert not hasattr(config, "RUNPOD_EMBEDDING_ENDPOINT_ID")
    assert config.HF_EMBEDDING_PROVIDER == "hf-inference"
    assert config.HF_EMBEDDING_DIMENSIONS == 1024
    assert config.CHUNK_TOKENIZER_MODEL == config.SELECTED_LLM_MODEL
    assert 300 <= config.CHUNK_MIN_TOKENS <= config.CHUNK_TARGET_TOKENS
    assert config.CHUNK_TARGET_TOKENS <= config.CHUNK_MAX_TOKENS <= 450
    assert 50 <= config.CHUNK_OVERLAP_TOKENS <= 75
    assert config.DIAGNOSTIC_GRADES == (10, 11)
    assert config.IMAGE_GEN_ENABLED is False
