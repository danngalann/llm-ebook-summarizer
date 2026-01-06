from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from constants import AI_BATCH_SIZE, ENABLE_CUDA_GRAPH

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
    max_num_seqs=32,  # Reduce from default 256 to lower memory usage during batching
#    gpu_memory_utilization=0.85,  # Reduce from default 0.9 to leave more headroom
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

def extract_notes(text):
    """Extract notes from a single chapter."""
    prompt = f"""
    Your task is to make notes from chapters of a book. You are given the text content of a chapter from a book, and you provide a short summary and notes from the chapter in Markdown.    
    
    CHAPTER TEXT:
    {text}
    END CHAPTER TEXT

    The summary must contain the following structure:
    - Title (extracted or inferred from text, always a # header)
    - Summary: A short summary of the content
    - Notes
      - Lessons (stuff that the chapter teaches, explains, or exposes)
      - Key points
      - Interesting quotes
    - Annecdotes or interesting comparisons

    Your response is pure markdown, without additional explanations sorrounding the notes and summary.
    """

    formatted_prompt = apply_chat_template(prompt)
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    response = outputs[0].outputs[0].text
    return response.strip()

def extract_notes_batch(texts):
    """Extract notes from multiple chapters in batches to manage memory.
    
    Args:
        texts (list[str]): List of chapter texts
        
    Returns:
        list[str]: List of extracted notes for each chapter
    """
    all_results = []
    
    # Process in chunks to avoid OOM
    for i in range(0, len(texts), AI_BATCH_SIZE):
        batch_texts = texts[i:i+AI_BATCH_SIZE]
        prompts = []
        
        for text in batch_texts:
            prompt = f"""
    Your task is to make notes from chapters of a book. You are given the text content of a chapter from a book, and you provide a short summary and notes from the chapter in Markdown.    
    
    CHAPTER TEXT:
    {text}
    END CHAPTER TEXT

    The summary must contain the following structure:
    - Title (extracted or inferred from text, always a # header)
    - Summary: A short summary of the content
    - Notes
      - Lessons (stuff that the chapter teaches, explains, or exposes)
      - Key points
      - Interesting quotes
    - Annecdotes or interesting comparisons

    Your response is pure markdown, without additional explanations sorrounding the notes and summary.
    """
            prompts.append(apply_chat_template(prompt))
        
        outputs = llm.generate(prompts, sampling_params)
        all_results.extend([output.outputs[0].text.strip() for output in outputs])
    
    return all_results

def translate_text(text, target_lang):
    """Translate a single text."""
    prompt = f"""
    Your task is to translate the following markdown text from its origional language to {target_lang}.
    You must keep the headings (#, ##, etc), lists and other markdown items just as they are.

    {text}
    """

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
            prompt = f"""
    Your task is to translate the following markdown text from its origional language to {target_lang}.
    You must keep the headings (#, ##, etc), lists and other markdown items just as they are.

    {text}
    """
            prompts.append(apply_chat_template(prompt))
        
        outputs = llm.generate(prompts, sampling_params)
        all_results.extend([output.outputs[0].text.strip() for output in outputs])
    
    return all_results
