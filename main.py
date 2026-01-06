import os
import argparse
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

from llm import extract_notes_batch
from tqdm import tqdm

OUTPUT_DIR = "output"

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

def extract_chapters(epub_path: str) -> None:
    """
    Extracts chapters from the given EPUB file and processes them in batches using vLLM.

    Args:
        epub_path (str): Path to the EPUB file.
    """
    MIN_WORD_COUNT = 200
    book: epub.EpubBook = epub.read_epub(epub_path)
    chapters = list(book.get_items_of_type(ITEM_DOCUMENT))
    
    # Prepare chapter data
    chapter_data = []
    for item in chapters:
        soup: BeautifulSoup = BeautifulSoup(item.get_content(), 'html.parser')
        chapter_text: str = soup.get_text(separator='\n', strip=True)
        words = chapter_text.split()
        
        if len(words) >= MIN_WORD_COUNT:
            chapter_data.append({
                'filename': item.get_name(),
                'text': ' '.join(words)
            })
    
    if not chapter_data:
        print("No chapters found that meet the minimum word count.")
        return
    
    print(f"Processing {len(chapter_data)} chapters in batch...")
    
    # Extract texts for batch processing
    texts = [chapter['text'] for chapter in chapter_data]
    
    # Process all chapters in one batch (vLLM handles internal batching efficiently)
    summaries = extract_notes_batch(texts)
    
    # Save results with progress bar
    for chapter, summary in tqdm(zip(chapter_data, summaries), total=len(chapter_data), desc="Saving chapters"):
        if summary:
            save_single_chapter(chapter['filename'], summary)

def main():
    parser = argparse.ArgumentParser(description="Process an EPUB book and extract chapter notes.")
    parser.add_argument("epub_file", help="Path to the EPUB file to be processed")
    args = parser.parse_args()

    ensure_output_dir_exists()
    
    extract_chapters(args.epub_file)

    print("Processing complete. All chapters have been processed and saved.")

if __name__ == "__main__":
    main()
