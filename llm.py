from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from constants import AI_BATCH_SIZE, ENABLE_CUDA_GRAPH, GPU_MEMORY_UTILIZATION
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

# Model configuration
MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Initialize LLM with 4-bit quantization
rope_theta = 1000000
original_max_position_embeddings = 32768
# Reduced factor to fit in available VRAM (estimated max: 57376)
factor = 1.75  # This gives us ~57344 tokens (32768 * 1.75)

hf_overrides = {
    "rope_parameters": {
        "rope_theta": rope_theta,
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_position_embeddings,
    },
    "max_model_len": 57376,  # Use estimated maximum based on available KV cache
}

llm = LLM(
    model=MODEL_ID,
    dtype="bfloat16",
    trust_remote_code=True,
    hf_overrides=hf_overrides,
    enforce_eager=not ENABLE_CUDA_GRAPH,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
)

# Sampling parameters for non-thinking mode (as per Qwen3 documentation)
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    max_tokens=4096,  # Increase max tokens to prevent truncation
)

def apply_chat_template(prompt: str) -> str:
    """
    Apply the chat template to the prompt with thinking mode disabled.
    
    Args:
        prompt (str): The user prompt
        
    Returns:
        str: The formatted text with chat template applied
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking mode
    )
    return text

def classify_chapters_batch(texts):
    """Classify chapters to determine if they are relevant for processing.
    
    This function runs the classification node on all chapters in batches.
    Chapters are considered relevant if the LLM output includes 'RELEVANT' (case-insensitive).
    
    Args:
        texts (list[str]): List of chapter texts
        
    Returns:
        list[str]: List of chapter texts that are classified as relevant
    """
    relevant_texts = []
    
    # Process in chunks to avoid OOM
    for i in range(0, len(texts), AI_BATCH_SIZE):
        batch_texts = texts[i:i+AI_BATCH_SIZE]
        
        # Classify all chapters in the batch
        classify_prompts = []
        for text in batch_texts:
            prompt = CLASSIFY_CHAPTER_PROMPT.format(text=text)
            classify_prompts.append(apply_chat_template(prompt))
        
        classifications = llm.generate(classify_prompts, sampling_params)
        classifications = [output.outputs[0].text.strip() for output in classifications]
        
        # Filter texts based on classification (case-insensitive, exact match)
        for j, classification in enumerate(classifications):
            if "relevant" == classification.lower():
                relevant_texts.append(batch_texts[j])
    
    return relevant_texts

def extract_notes_batch(texts):
    """Extract notes from multiple chapters using multi-node processing.
    
    This function processes each chapter through multiple task-specific nodes:
    1. Summary - generates main chapter title and summary
    2. Lessons - extracts lessons learned
    3. Key points - extracts key points
    4. Quotes - extracts interesting quotes
    5. Anecdotes - extracts interesting anecdotes or comparisons
    6. Resources - extracts any mentioned resources or further reading
    
    The results are then aggregated into a structured markdown format.
    
    Args:
        texts (list[str]): List of chapter texts
        
    Returns:
        list[str]: List of aggregated summaries for each chapter
    """
    all_results = []
    
    # Process in chunks to avoid OOM
    for i in range(0, len(texts), AI_BATCH_SIZE):
        batch_texts = texts[i:i+AI_BATCH_SIZE]
        
        # Process each node sequentially to manage GPU memory
        # But process all chapters in the batch at once for each node
        
        # Node 1: Extract summary (includes title)
        summary_prompts = []
        for text in batch_texts:
            prompt = EXTRACT_SUMMARY_PROMPT.format(text=text)
            summary_prompts.append(apply_chat_template(prompt))
        summaries = llm.generate(summary_prompts, sampling_params)
        summaries = [output.outputs[0].text.strip() for output in summaries]
        
        # Node 2: Extract lessons
        lessons_prompts = []
        for text in batch_texts:
            prompt = EXTRACT_LESSONS_PROMPT.format(text=text)
            lessons_prompts.append(apply_chat_template(prompt))
        lessons = llm.generate(lessons_prompts, sampling_params)
        lessons = [output.outputs[0].text.strip() for output in lessons]
        
        # Node 3: Extract key points
        keypoints_prompts = []
        for text in batch_texts:
            prompt = EXTRACT_KEYPOINTS_PROMPT.format(text=text)
            keypoints_prompts.append(apply_chat_template(prompt))
        keypoints = llm.generate(keypoints_prompts, sampling_params)
        keypoints = [output.outputs[0].text.strip() for output in keypoints]
        
        # Node 4: Extract quotes
        quotes_prompts = []
        for text in batch_texts:
            prompt = EXTRACT_QUOTES_PROMPT.format(text=text)
            quotes_prompts.append(apply_chat_template(prompt))
        quotes = llm.generate(quotes_prompts, sampling_params)
        quotes = [output.outputs[0].text.strip() for output in quotes]
        
        # Node 5: Extract anecdotes
        anecdotes_prompts = []
        for text in batch_texts:
            prompt = EXTRACT_ANECDOTES_PROMPT.format(text=text)
            anecdotes_prompts.append(apply_chat_template(prompt))
        anecdotes = llm.generate(anecdotes_prompts, sampling_params)
        anecdotes = [output.outputs[0].text.strip() for output in anecdotes]
        
        # Node 6: Extract resources
        resources_prompts = []
        for text in batch_texts:
            prompt = EXTRACT_RESOURCES_PROMPT.format(text=text)
            resources_prompts.append(apply_chat_template(prompt))
        resources = llm.generate(resources_prompts, sampling_params)
        resources = [output.outputs[0].text.strip() for output in resources]
        
        # Aggregate results for each chapter in the batch
        for j in range(len(batch_texts)):
            aggregated = _aggregate_node_results(
                summaries[j],
                lessons[j],
                keypoints[j],
                quotes[j],
                anecdotes[j],
                resources[j]
            )
            all_results.append(aggregated)
    
    return all_results


def _aggregate_node_results(summary, lessons, keypoints, quotes, anecdotes, resources):
    """Aggregate results from all nodes into a structured markdown format.
    
    Args:
        summary (str): Summary with title (# header)
        lessons (str): Lessons content
        keypoints (str): Key points content
        quotes (str): Quotes content
        anecdotes (str): Anecdotes content
        resources (str): Resources content
        
    Returns:
        str: Aggregated markdown content
    """
    # Start with the summary (which includes the title as # header)
    parts = [summary]
    
    # Add each section with hardcoded headers
    if lessons.strip():
        parts.append("\n# Lessons\n" + lessons)
    
    if keypoints.strip():
        parts.append("\n# Key Points\n" + keypoints)
    
    if quotes.strip():
        parts.append("\n# Quotes\n" + quotes)
    
    if anecdotes.strip():
        parts.append("\n# Anecdotes\n" + anecdotes)
    
    # Skip resources section if the special message is returned (case insensitive)
    if resources.strip() and resources.strip().lower() != "no resources mentioned.":
        parts.append("\n# Resources\n" + resources)
    
    return "\n".join(parts)

def translate_text(text, target_lang):
    """Translate a single text."""
    prompt = TRANSLATE_TEXT_PROMPT.format(text=text, target_lang=target_lang)

    formatted_prompt = apply_chat_template(prompt)
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    response = outputs[0].outputs[0].text
    return response.strip()

def translate_text_batch(texts, target_lang):
    """Translate multiple texts in batches to manage memory.
    
    Args:
        texts (list[str]): List of texts to translate
        target_lang (str): Target language
        
    Returns:
        list[str]: List of translated texts
    """
    all_results = []
    
    # Process in chunks to avoid OOM
    for i in range(0, len(texts), AI_BATCH_SIZE):
        batch_texts = texts[i:i+AI_BATCH_SIZE]
        prompts = []
        
        for text in batch_texts:
            prompt = TRANSLATE_TEXT_PROMPT.format(text=text, target_lang=target_lang)
            prompts.append(apply_chat_template(prompt))
        
        outputs = llm.generate(prompts, sampling_params)
        all_results.extend([output.outputs[0].text.strip() for output in outputs])
    
    return all_results
