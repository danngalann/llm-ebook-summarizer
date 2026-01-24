"""
LLM client for book summarization using Ollama.

WHY OLLAMA INSTEAD OF vLLM:
This project performs offline, batch summarization with low concurrency (1-2 active sequences).
vLLM aggressively preallocates KV cache for high-throughput batching, which limits usable model size
on a 24 GB GPU to ~7B-8B quantized models even with 4-bit quantization.

Ollama uses conservative memory allocation optimized for single-sequence inference, allowing
32B-class quantized models (e.g., qwen3:30b-a3b-instruct-2507-q4_K_M) to run on the same hardware. For this workload,
Ollama provides significantly better output quality with comparable end-to-end runtime and
much lower operational complexity.

This design choice prioritizes summary quality over batching throughput, which is appropriate
for offline batch processing on single-GPU hardware.
"""

import ollama
from typing import List
from tqdm import tqdm

from constants import OLLAMA_MODEL, OLLAMA_BASE_URL
from prompts import (
    CLASSIFY_CHAPTER_PROMPT,
    EXTRACT_SUMMARY_PROMPT,
    EXTRACT_LESSONS_PROMPT,
    EXTRACT_KEYPOINTS_PROMPT,
    EXTRACT_QUOTES_PROMPT,
    EXTRACT_ANECDOTES_PROMPT,
    EXTRACT_RESOURCES_PROMPT,
    TRANSLATE_TEXT_PROMPT
)

# Initialize Ollama client
client = ollama.Client(host=OLLAMA_BASE_URL)

# Generation parameters
GENERATION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "num_predict": 8192,  # Max tokens to generate (output length) - increased for longer summaries/translations
    "num_ctx": 32768,  # Context window size (input + output) - Qwen3 supports up to 262k, but 32k is sufficient here
}


def _generate(prompt: str) -> str:
    """
    Generate a response from the Ollama model.
    
    Args:
        prompt: The user prompt
        
    Returns:
        The generated text response
        
    Raises:
        Exception: If Ollama connection or generation fails
    """
    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=GENERATION_OPTIONS,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        raise Exception(f"Ollama generation failed: {e}")

def classify_chapters_batch(texts: List[str]) -> List[str]:
    """
    Classify chapters to determine if they are relevant for processing.
    
    Processes chapters sequentially (Ollama is optimized for single-sequence inference).
    Chapters are considered relevant if the LLM output includes 'RELEVANT' (case-insensitive).
    
    Args:
        texts: List of chapter texts
        
    Returns:
        List of chapter texts that are classified as relevant
    """
    relevant_texts = []
    
    for text in tqdm(texts, desc="Classifying chapters", unit="chapter"):
        prompt = CLASSIFY_CHAPTER_PROMPT.format(text=text)
        classification = _generate(prompt)
        
        # Filter texts based on classification (case-insensitive)
        if "relevant" == classification.lower():
            relevant_texts.append(text)
    
    return relevant_texts

def extract_notes_batch(texts: List[str]) -> List[str]:
    """
    Extract notes from multiple chapters using multi-node processing.
    
    This function processes each chapter through multiple task-specific nodes:
    1. Summary - generates main chapter title and summary
    2. Lessons - extracts lessons learned
    3. Key points - extracts key points
    4. Quotes - extracts interesting quotes
    5. Anecdotes - extracts interesting anecdotes or comparisons
    6. Resources - extracts any mentioned resources or further reading
    
    The results are then aggregated into a structured markdown format.
    
    Args:
        texts: List of chapter texts
        
    Returns:
        List of aggregated summaries for each chapter
    """
    all_results = []
    
    # Process each chapter sequentially through all nodes
    for text in tqdm(texts, desc="Processing chapters", unit="chapter"):
        # Node 1: Extract summary (includes title)
        prompt = EXTRACT_SUMMARY_PROMPT.format(text=text)
        summary = _generate(prompt)
        
        # Node 2: Extract lessons
        prompt = EXTRACT_LESSONS_PROMPT.format(text=text)
        lessons = _generate(prompt)
        
        # Node 3: Extract key points
        prompt = EXTRACT_KEYPOINTS_PROMPT.format(text=text)
        keypoints = _generate(prompt)
        
        # Node 4: Extract quotes
        prompt = EXTRACT_QUOTES_PROMPT.format(text=text)
        quotes = _generate(prompt)
        
        # Node 5: Extract anecdotes
        prompt = EXTRACT_ANECDOTES_PROMPT.format(text=text)
        anecdotes = _generate(prompt)
        
        # Node 6: Extract resources
        prompt = EXTRACT_RESOURCES_PROMPT.format(text=text)
        resources = _generate(prompt)
        
        # Aggregate results
        aggregated = _aggregate_node_results(
            summary,
            lessons,
            keypoints,
            quotes,
            anecdotes,
            resources
        )
        all_results.append(aggregated)
    
    return all_results


def _aggregate_node_results(
    summary: str,
    lessons: str,
    keypoints: str,
    quotes: str,
    anecdotes: str,
    resources: str
) -> str:
    """
    Aggregate results from all nodes into a structured markdown format.
    
    Args:
        summary: Summary with title (# header)
        lessons: Lessons content
        keypoints: Key points content
        quotes: Quotes content
        anecdotes: Anecdotes content
        resources: Resources content
        
    Returns:
        Aggregated markdown content
    """
    # Start with the summary (which includes the title as # header)
    parts = [summary]
    
    # Add each section with hardcoded headers
    if lessons.strip():
        parts.append("\n## Lessons\n" + lessons)
    
    if keypoints.strip():
        parts.append("\n## Key Points\n" + keypoints)
    
    if quotes.strip():
        parts.append("\n## Quotes\n" + quotes)
    
    if anecdotes.strip():
        parts.append("\n## Anecdotes\n" + anecdotes)
    
    # Skip resources section if the special message is returned (case insensitive)
    if resources.strip() and resources.strip().lower() != "no resources mentioned.":
        parts.append("\n## Resources\n" + resources)
    
    return "\n".join(parts)

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate a single text.
    
    Args:
        text: Text to translate
        target_lang: Target language
        
    Returns:
        Translated text
    """
    prompt = TRANSLATE_TEXT_PROMPT.format(text=text, target_lang=target_lang)
    return _generate(prompt)


def translate_text_batch(texts: List[str], target_lang: str) -> List[str]:
    """
    Translate multiple texts sequentially.
    
    Args:
        texts: List of texts to translate
        target_lang: Target language
        
    Returns:
        List of translated texts
    """
    all_results = []
    
    for text in tqdm(texts, desc=f"Translating to {target_lang}", unit="text"):
        prompt = TRANSLATE_TEXT_PROMPT.format(text=text, target_lang=target_lang)
        translation = _generate(prompt)
        all_results.append(translation)
    
    return all_results
