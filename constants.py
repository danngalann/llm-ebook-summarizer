"""Application-wide constants."""

# Minimum word count for a chapter to be included in processing
MIN_WORD_COUNT = 200

# Default output directory for processed chapters
OUTPUT_DIR = "output"

AI_BATCH_SIZE = 50  # Number of chapters to process in a single batch

# Enable CUDA Graphs for vLLM, consumes more GPU memory but speeds up inference
ENABLE_CUDA_GRAPH = True