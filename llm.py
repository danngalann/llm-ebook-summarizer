"""Book summarization with local Ollama or OpenRouter."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from dotenv import load_dotenv
import httpx
import ollama
from tqdm import tqdm

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

load_dotenv(Path(__file__).with_name(".env"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL")
REMOTE_CONCURRENCY = int(os.getenv("REMOTE_CONCURRENCY", "6"))

parsed_url = urlparse(LLM_BASE_URL)
if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
    raise ValueError("LLM_BASE_URL must be an HTTP(S) URL")

IS_OPENROUTER = parsed_url.hostname == "openrouter.ai"
if IS_OPENROUTER:
    if parsed_url.scheme != "https":
        raise ValueError("OpenRouter requires an HTTPS URL")
    if not LLM_API_KEY:
        raise ValueError("LLM_API_KEY is required for OpenRouter")
    if REMOTE_CONCURRENCY < 1:
        raise ValueError("REMOTE_CONCURRENCY must be positive")
    base_url = LLM_BASE_URL.rstrip("/")
    if parsed_url.path in ("", "/"):
        base_url += "/api/v1"
    elif parsed_url.path.rstrip("/") != "/api/v1":
        raise ValueError("OpenRouter URL must end in /api/v1")
    client = httpx.Client(
        base_url=base_url + "/",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        timeout=120,
        limits=httpx.Limits(max_connections=REMOTE_CONCURRENCY),
    )
else:
    client = ollama.Client(host=LLM_BASE_URL)

MODEL = LLM_MODEL or (None if IS_OPENROUTER else "qwen3:30b-a3b-instruct-2507-q4_K_M")
CONCURRENCY = REMOTE_CONCURRENCY if IS_OPENROUTER else 1

# Ollama only: OpenRouter models have their own context and output limits.
GENERATION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "num_predict": 8192,  # Max tokens to generate (output length) - increased for longer summaries/translations
    "num_ctx": 32768,  # Context window size (input + output) - Qwen3 supports up to 262k, but 32k is sufficient here
}


def _generate(prompt: str) -> str:
    """Generate a response from the configured model."""
    if IS_OPENROUTER:
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": GENERATION_OPTIONS["temperature"],
            "top_p": GENERATION_OPTIONS["top_p"],
        }
        if MODEL:
            request["model"] = MODEL
        response = client.post("chat/completions", json=request)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("OpenRouter response exceeded the model's output limit")
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter returned an empty text response")
        return content.strip()
    else:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options=GENERATION_OPTIONS,
        )
        return response["message"]["content"].strip()

def classify_chapters_batch(texts: List[str]) -> List[str]:
    """
    Classify chapters to determine if they are relevant for processing.
    
    Uses sequential requests locally and concurrent requests on OpenRouter.
    Chapters are considered relevant if the LLM output is 'RELEVANT' (case-insensitive).
    
    Args:
        texts: List of chapter texts
        
    Returns:
        List of chapter texts that are classified as relevant
    """
    def classify(text: str) -> bool:
        prompt = CLASSIFY_CHAPTER_PROMPT.format(text=text)
        classification = _generate(prompt)
        return classification.lower() == "relevant"

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        classifications = list(tqdm(
            executor.map(classify, texts), total=len(texts),
            desc="Classifying chapters", unit="chapter",
        ))
    return [text for text, relevant in zip(texts, classifications) if relevant]

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
    prompts = (
        EXTRACT_SUMMARY_PROMPT,
        EXTRACT_LESSONS_PROMPT,
        EXTRACT_KEYPOINTS_PROMPT,
        EXTRACT_QUOTES_PROMPT,
        EXTRACT_ANECDOTES_PROMPT,
        EXTRACT_RESOURCES_PROMPT,
    )

    if not IS_OPENROUTER:
        return [
            _aggregate_node_results(*(
                _generate(template.format(text=text)) for template in prompts
            ))
            for text in tqdm(texts, desc="Processing chapters", unit="chapter")
        ]

    def generate_task(task: tuple[str, str]) -> str:
        text, template = task
        return _generate(template.format(text=text))

    tasks = ((text, template) for text in texts for template in prompts)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        responses = iter(executor.map(generate_task, tasks))
        return [
            _aggregate_node_results(*(next(responses) for _ in prompts))
            for _ in tqdm(texts, desc="Processing chapters", unit="chapter")
        ]


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
    Translate multiple texts; use concurrent requests on OpenRouter.
    
    Args:
        texts: List of texts to translate
        target_lang: Target language
        
    Returns:
        List of translated texts
    """
    def translate(text: str) -> str:
        prompt = TRANSLATE_TEXT_PROMPT.format(text=text, target_lang=target_lang)
        return _generate(prompt)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        return list(tqdm(
            executor.map(translate, texts), total=len(texts),
            desc=f"Translating to {target_lang}", unit="text",
        ))
