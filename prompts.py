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

EXTRACT_SUMMARY_PROMPT = """
Your task is to make a summary from a chapter of a book. You are given the text content of the chapter, and you provide a summary from the chapter in Markdown.

CHAPTER TEXT:
{text}
END CHAPTER TEXT

The summary must contain the following structure:
- Title (extracted or inferred from text, always a # header)
- Summary: A summary of the content that captures the main ideas and themes.

Rules:
- Your response is pure markdown, without additional explanations sorrounding the summary.
- The scope of your task is only to provide a general overview summary. Extracting key points, lessons, quotes, or annecdotes is not required.
"""

EXTRACT_LESSONS_PROMPT = """
Your task is to extract lessons from chapters of a book. You are given the text content of a chapter from a book, and you provide an overview of lessons learned from the chapter in Markdown.

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Rules:
- Your response is pure markdown, without additional explanations sorrounding the lessons overview.
- If the chapter already contains a recap section or similar, you can use that as a basis for the lessons overview.
- If the chapter contains lists, tables, etc you are allowed to use those in your response.
- You cannot use headings
"""

EXTRACT_KEYPOINTS_PROMPT = """
Your task is to extract key points from chapters of a book. You are given the text content of a chapter from a book, and you provide an overview of key points from the chapter in Markdown

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Your response must follow a structure where key points are presented as a bulleted list with concise explanations.

Rules:
- Your response is pure markdown, without additional explanations sorrounding the key points overview.
- You cannot use headings
"""

EXTRACT_QUOTES_PROMPT = """
Your task is to extract interesting quotes from chapters of a book. You are given the text content of a chapter from a book, and you provide a list of interesting quotes from the chapter in Markdown.
The idea is for the quotes to represent important or thought-provoking statements made about ideas in the chapter.

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Rules:
- Your response is pure markdown, without additional explanations sorrounding the quotes.
- Present the quotes as a bulleted list.
- You cannot use headings
- Return no more than the top 5 most relevant quotes.
"""

EXTRACT_ANECDOTES_PROMPT = """
Your task is to extract interesting anecdotes or comparisons from chapters of a book. You are given the text content of a chapter from a book, and you provide a list of interesting anecdotes or comparisons from the chapter in Markdown.
The idea is for the anecdotes or comparisons to represent memorable stories or analogies made about ideas in the chapter.

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Rules:
- Your response is pure markdown, without additional explanations sorrounding the anecdotes or comparisons.
- Present the anecdotes or comparisons as series of sections with level 2 headings (##) for each anecdote or comparison, with a short summary under each heading.
"""

EXTRACT_RESOURCES_PROMPT = """
Your task is to extract a list of resources (books, articles, papers, websites, etc) referenced in the chapter of a book. You are given the text content of a chapter from a book, and you provide a list of resources mentioned in the chapter in Markdown.

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Rules:
- Your response is pure markdown, without additional explanations sorrounding the resources.
- Present the resources as a bulleted list with the resource name or link and a short description of what the resource is about, if available.
- You cannot use headings
- If the chapter does not mention any resources, respond with "No resources mentioned."
"""

TRANSLATE_TEXT_PROMPT = """
Your task is to translate the following markdown text from its origional language to {target_lang}.
You must keep the headings (#, ##, etc), lists and other markdown items just as they are.

{text}
"""
