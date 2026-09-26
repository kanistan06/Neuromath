"""Backward-compatible aliases for the former Hugging Face text-inference module.

Text generation has moved to RunPod Serverless.  This shim keeps older internal
imports from breaking while routing every generation/review call to RunPod.
"""

from rag.runpod_inference import (  # noqa: F401
    RUNPOD_INFERENCE_IMPLEMENTATION,
    RUNPOD_MCQ_GENERATION_IMPLEMENTATION,
    runpod_json_generation,
    runpod_mcq_generation,
    runpod_mcq_review,
)

# Compatibility names for older callers. No Hugging Face text inference occurs.
HF_INFERENCE_IMPLEMENTATION = RUNPOD_INFERENCE_IMPLEMENTATION
HF_MCQ_GENERATION_IMPLEMENTATION = RUNPOD_MCQ_GENERATION_IMPLEMENTATION
hf_json_generation = runpod_json_generation
hf_mcq_generation = runpod_mcq_generation
hf_mcq_review = runpod_mcq_review
