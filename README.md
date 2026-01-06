# LLM Ebook Summarizer

A tool for extracting and summarizing chapters from ebooks using LLMs. Perfect for creating concise study notes, book summaries, or content overviews.

## Features

### Multi-Format Support
- **EPUB files** — Extract and summarize chapters from EPUB ebooks
- **PDF files** — Process PDFs with table of contents
- **Nested chapters** — Handles hierarchical book structures (parts, chapters, sections)
- **Context preservation** — Includes parent section introductions when summarizing nested content

### AI-Powered Processing
- Generates structured markdown notes for each chapter
- Extracts key insights: summaries, lessons, quotes, and anecdotes
- Batch processing for efficient summarization
- Maintains 200-word minimum filter for quality content

### Translation
- Translate all generated summaries to any language
- Batch translation for speed and consistency
- Preserves markdown formatting

### Output Management
- Individual markdown files per section
- Sequential numbering with descriptive filenames
- Merge utility to combine into single file

## Usage

### 1. Summarize an Ebook

Process an EPUB or PDF file to generate chapter summaries:

```bash
python main.py path/to/your-book.epub
```

```bash
python main.py path/to/your-book.pdf
```

**Output:** Individual markdown files saved to `output/` directory
- Format: `001_section_name.md`, `002_introduction.md`, etc.

**Requirements:**
- For PDFs: Must have a table of contents/outline
- Minimum 200 words per section

### 2. Translate Summaries

Translate all generated summaries to another language:

```bash
python translate.py es
```

Replace `es` with your target language code:
- `es` — Spanish
- `fr` — French
- `de` — German
- `pt` — Portuguese
- `ja` — Japanese
- etc.

**Output:** Translated files saved to `translations/` directory
- Format: `001_chapter_name_es.md`, `002_introduction_es.md`, etc.

### 3. Merge into Single File

Combine all chapter summaries into one consolidated markdown file:

```bash
python merge_markdowns.py output/
```

For translated versions:

```bash
python merge_markdowns.py translations/
```

**Output:** `book.md` file containing all chapters in alphabetical order

## Workflow Example

Complete workflow for processing a book:

```bash
# 1. Summarize the book
python main.py "Designing Data-Intensive Applications.pdf"

# 2. (Optional) Translate to Spanish
python translate.py es

# 3. (Optional) Merge into single file
python merge_markdowns.py output/
```

## Output Structure

### For Books with Nested Chapters

Given a PDF with structure:
```
Part I: Foundations
├─ Chapter 1: Introduction
│  ├─ Section 1.1: Basic Concepts
│  └─ Section 1.2: Core Principles
└─ Chapter 2: Getting Started
```

The tool:
1. Identifies leaf sections (the innermost chapters/sections)
2. For each leaf, includes introductory text from ancestor sections
   - Only the **first leaf** under each ancestor receives that ancestor's intro text
   - This prevents sibling sections from getting duplicate introductory content
3. Generates one summary file per leaf section

For example, when summarizing "Section 1.1: Basic Concepts" (first subsection of Chapter 1):
- Receives intro from "Part I: Foundations" (before Chapter 1 starts)
- Receives intro from "Chapter 1: Introduction" (before Section 1.1 starts)
- Receives full content of "Section 1.1: Basic Concepts"

But "Section 1.2: Core Principles" (sibling of 1.1) only receives:
- Receives full content of "Section 1.2: Core Principles"
- Does NOT receive intro from Part I or Chapter 1 (to avoid repetition)

### Filename Convention

Files are named with sequential numbers plus sanitized titles:
- `003_section_1_1_basic_concepts.md`
- `004_section_1_2_core_principles.md`
- `005_chapter_2_getting_started.md`

## Installation

Install dependencies:

```bash
uv sync
```

## Notes

- Processing time depends on book length and available GPU/CPU resources
- Generated summaries include: title, summary, notes, lessons, key points, quotes, and anecdotes
- All markdown files use UTF-8 encoding
- Scanned PDFs (image-only) are not supported — text-based PDFs only
