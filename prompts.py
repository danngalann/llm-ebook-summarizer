# Prompt templates with placeholders for string.format()

EXTRACT_NOTES_PROMPT = """
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

TRANSLATE_TEXT_PROMPT = """
Your task is to translate the following markdown text from its origional language to {target_lang}.
You must keep the headings (#, ##, etc), lists and other markdown items just as they are.

{text}
"""
