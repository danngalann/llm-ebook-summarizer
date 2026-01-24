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
Your task is to write a clear, compact summary of a book chapter.

The summary should read as an integrated overview of the chapter's content, not as a commentary about the chapter.
Focus on what the chapter says, not on describing what it does.

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Structure:
- Title (extracted or inferred from the text, always a # header)
- Summary: 1-3 short paragraphs forming a coherent narrative overview

Style rules:
- Write in a neutral, assertive, non-academic tone.
- Avoid metadiscourse (do NOT use phrases like "this chapter explores", "the text discusses", "the author explains").
- Prefer direct statements over explanatory framing.
- Maintain logical flow between ideas; avoid list-like phrasing.
- Do not include lessons, advice, key points, quotes, or anecdotes.

Format rules:
- Pure Markdown only.
"""

EXTRACT_LESSONS_PROMPT = """
Your task is to extract LESSONS from a book chapter.

A lesson is a generalized insight, principle, or takeaway that can be applied beyond the specific chapter context.
Lessons answer the question: "What should a reader learn and carry forward?"

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Rules:
- Produce ONLY lessons, not summaries or factual descriptions.
- Each lesson must be abstracted beyond specific examples, people, institutions, or anecdotes from the chapter.
- Avoid chapter-specific details (names, statistics, events, studies).
- Frame lessons as transferable insights that could apply in other contexts.
- Use a bulleted list.
- Pure Markdown only.
- Do NOT use headings.
"""

EXTRACT_KEYPOINTS_PROMPT = """
Your task is to extract KEY POINTS from a book chapter.

Key points are the main factual ideas, arguments, or claims presented in the chapter.
Key points answer the question: "What information does this chapter present?"

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Rules:
- Stay close to the chapter content; do not generalize beyond what is stated.
- Include important facts, arguments, examples, and explanations from the chapter.
- It is acceptable to mention specific concepts, institutions, research areas, or examples if they appear in the text.
- Present key points as a bulleted list with concise explanations.
- Pure Markdown only.
- Do NOT use headings.
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
Your task is to extract ANECDOTES and CONCRETE COMPARISONS from a book chapter.

An anecdote is a specific story, event, scene, or personal account.
A comparison is a concrete analogy explicitly used in the text to explain an idea.

These must be narrative or illustrative elements, not interpretations or summaries.

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Selection criteria:
- Choose the most memorable or illustrative stories or examples.
- Prefer concrete narratives over abstract explanations.
- Do NOT generalize or interpret beyond what happens in the anecdote.

Style rules:
- Do NOT summarize the chapter.
- Do NOT explain themes, lessons, or implications.
- Do NOT use phrases like "the chapter highlights", "this shows", "this illustrates".
- Stay close to the original narrative content.

Format:
- Pure Markdown only.
- Present up to 3 anecdotes.
- Each anecdote must have:
  - A level 2 heading (##) with a short descriptive title.
  - A concise narrative summary focused on what happened.
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

CLASSIFY_CHAPTER_PROMPT = """
Your task is to classify whether a chapter from a book is relevant for processing based on its content. You are given the text content of a chapter from a book, and you must respond with either "RELEVANT" or "NOT RELEVANT".

CHAPTER TEXT:
{text}
END CHAPTER TEXT

Rules:
- If the chapter contains meaningful content related to the book's subject matter, respond with "RELEVANT".
- If the chapter is mostly filler, preface, introduction, table of contents, acknowledgements, praise, or similar non-content, respond with "NOT RELEVANT".
- Your response must be exactly either "RELEVANT" or "NOT RELEVANT", without additional explanations.
"""

TRANSLATE_TEXT_PROMPT = """
Your task is to translate the following markdown text from its origional language to {target_lang}.
You must keep the headings (#, ##, etc), lists and other markdown items just as they are.

{text}
"""
