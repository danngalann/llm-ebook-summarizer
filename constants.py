"""Application-wide constants."""

# Minimum word count for a chapter to be included in processing
MIN_WORD_COUNT = 200

# Default output directory for processed chapters
OUTPUT_DIR = "output"

# Ollama configuration
OLLAMA_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"  # 32B quantized model for high-quality summaries
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama server URL