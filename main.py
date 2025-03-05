import os
import argparse
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from typing import Optional

from llm import extract_notes
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

def process_text(chapter_filename: str, text: str) -> None:
    """
    Processes the chapter text, generates notes, and saves them immediately if criteria are met.
    
    Args:
        chapter_filename (str): The filename or identifier for the chapter.
        text (str): The extracted text content of the chapter.
    """
    MIN_WORD_COUNT = 200
    words = text.split()

    if len(words) >= MIN_WORD_COUNT:
        extracted_summary: Optional[str] = extract_notes(" ".join(words))
        if extracted_summary:
            print(f"Processed summary for {chapter_filename}")
            save_single_chapter(chapter_filename, extracted_summary)

def extract_chapters(epub_path: str) -> None:
    """
    Extracts chapters from the given EPUB file and processes each chapter sequentially,
    with a progress bar indicating the process.

    Args:
        epub_path (str): Path to the EPUB file.
    """
    book: epub.EpubBook = epub.read_epub(epub_path)
    # Retrieve list of document items to determine total chapters for tqdm
    chapters = list(book.get_items_of_type(ITEM_DOCUMENT))

    # Wrap the chapters list in tqdm for a progress bar
    for item in tqdm(chapters, desc="Processing Chapters"):
        soup: BeautifulSoup = BeautifulSoup(item.get_content(), 'html.parser')
        chapter_text: str = soup.get_text(separator='\n', strip=True)
        process_text(item.get_name(), chapter_text)

def main():
    parser = argparse.ArgumentParser(description="Process an EPUB book and extract chapter notes.")
    parser.add_argument("epub_file", help="Path to the EPUB file to be processed")
    args = parser.parse_args()

    ensure_output_dir_exists()
    
    extract_chapters(args.epub_file)

    print("Processing complete. All chapters have been processed and saved.")

if __name__ == "__main__":
    main()
