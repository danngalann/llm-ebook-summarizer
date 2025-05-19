from langchain_ollama import OllamaLLM
import re

LLMS = {
    'qwen': 'qwen2.5:32b',
    'qwen3-a3b': 'qwen3:30b-a3b',
    'llama3.2': 'llama3.2:latest',
    'llama3.1': 'llama3.1:8b',
    'deepseek-r1': 'deepseek-r1:32b',
}

llm = OllamaLLM(model=LLMS['qwen3-a3b'], temperature=0.3, top_k=10, top_p=0.5)

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
    - Title (extracted or inferred from text, always a # header)
    - Summary: A short summary of the content
    - Notes
      - Lessons (stuff that the chapter teaches, explains, or exposes)
      - Key points
      - Interesting quotes
    - Annecdotes or interesting comparisons

    Your response is pure markdown, without additional explanations sorrounding the notes and summary.
    /no_think
    """

    response = llm.invoke(prompt)
    response = remove_thinking(response)

    return response.strip()

def translate_text(text, target_lang):
  prompt = f"""
  Your task is to translate the following markdown text from its origional language to {target_lang}.
  You must keep the headings (#, ##, etc), lists and other markdown items just as they are.

  /no_think

  {text}
  """

  response = llm.invoke(prompt)
  response = remove_thinking(response)

  return response.strip()