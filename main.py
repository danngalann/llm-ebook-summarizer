import os
import argparse
from tqdm import tqdm

from llm import extract_notes_batch
from epub_extractor import extract_epub_content
from pdf_extractor import extract_pdf_content
from constants import MIN_WORD_COUNT, OUTPUT_DIR

def ensure_output_dir_exists(output_dir: str = OUTPUT_DIR) -> None:
    """
    Ensures the output directory exists.
    """
    os.makedirs(output_dir, exist_ok=True)

def save_single_chapter(chapter_filename: str, notes: str, output_dir: str = OUTPUT_DIR) -> None:
    """
    Saves the processed notes for a single chapter into a Markdown file.

    Args:
        chapter_filename (str): The filename or identifier for the chapter.
        notes (str): The processed notes to save.
        output_dir (str): The directory where the Markdown file will be saved.
    """
    base_name = os.path.splitext(os.path.basename(chapter_filename))[0]
    md_filename = os.path.join(output_dir, f"{base_name}.md")

    with open(md_filename, "w", encoding="utf-8") as md_file:
        md_file.write(notes)

def extract_chapters(file_path: str) -> None:
    """
    Extracts chapters from the given EPUB or PDF file and processes them in batches using vLLM.
    For PDFs with nested TOC, processes leaf nodes and merges summaries hierarchically.

    Args:
        file_path (str): Path to the EPUB or PDF file.
    
    Raises:
        ValueError: If file format is unsupported or PDF has no chapters.
    """
    # Detect file type and extract content
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.epub':
        print(f"Detected EPUB file: {file_path}")
        chapter_data = extract_epub_content(file_path, MIN_WORD_COUNT)
    elif file_ext == '.pdf':
        print(f"Detected PDF file: {file_path}")
        chapter_data = extract_pdf_content(file_path, MIN_WORD_COUNT)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. "
            "Only .epub and .pdf files are supported."
        )
    
    if not chapter_data:
        print("No chapters found that meet the minimum word count.")
        return
    
    print(f"Processing {len(chapter_data)} sections in batch...")
    
    # Extract texts for batch processing
    if file_ext == '.pdf':
        texts = [item['text'] for item in chapter_data]
    else:
        texts = [chapter['text'] for chapter in chapter_data]
    
    # Process all sections in one batch (vLLM handles internal batching efficiently)
    summaries = extract_notes_batch(texts)
    
    # Save summaries
    print("Saving section summaries...")
    if file_ext == '.pdf':
        for item, summary in tqdm(zip(chapter_data, summaries), total=len(chapter_data), desc="Saving sections"):
            if summary:
                save_single_chapter(item['node'].get_filename(), summary)
    else:
        for chapter, summary in tqdm(zip(chapter_data, summaries), total=len(chapter_data), desc="Saving chapters"):
            if summary:
                save_single_chapter(chapter['filename'], summary)
    
    print(f"Saved {len(chapter_data)} section summaries.")

def main():
    parser = argparse.ArgumentParser(description="Process an EPUB or PDF book and extract chapter notes.")
    parser.add_argument("ebook_file", help="Path to the EPUB or PDF file to be processed")
    args = parser.parse_args()

    ensure_output_dir_exists()
    
    extract_chapters(args.ebook_file)

    print("Processing complete. All chapters have been processed and saved.")

if __name__ == "__main__":
    main()
