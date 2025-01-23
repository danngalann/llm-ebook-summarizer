from langchain_ollama import OllamaLLM
import re

LLMS = {
    'qwen': 'qwen2.5:32b',
    'llama3.2': 'llama3.2:latest',
    'llama3.1': 'llama3.1:8b',
    'deepseek-r1': 'deepseek-r1:32b',
}

llm = OllamaLLM(model=LLMS['deepseek-r1'], temperature=0.3, top_k=10, top_p=0.5)

def remove_thinking(text):
  """
  Remove every <think> tag from the text and its content
  """
  return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def extract_notes(text):
    prompt = f"""
    Your task is to make notes from chapters of a book. You are given the text content of a chapter from a book, and you provide a short summary and notes from the chapter in Markdown.    
    
    CHAPTER TEXT:
    {text}
    END CHAPTER TEXT

    The summary must contain the following structure:
    - Title (extracted or inferred from text)
    - Summary: A short summary of the content
    - Notes
      - Key points
      - Interesting quotes
    - Annecdotes or interesting comparisons

    Your response is pure markdown, without additional explanations sorrounding the notes and summary.
    """

    response = llm.invoke(prompt)
    response = remove_thinking(response)

    return response.strip()