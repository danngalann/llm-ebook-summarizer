"""EPUB content extraction module."""

import os
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

from utils import sanitize_filename


def extract_epub_content(epub_path: str, min_word_count: int = 200) -> list:
    """
    Extracts chapters from an EPUB file.
    
    Args:
        epub_path (str): Path to the EPUB file.
        min_word_count (int): Minimum word count for a chapter to be included.
    
    Returns:
        list: List of dictionaries with 'filename' and 'text' keys.
    """
    book: epub.EpubBook = epub.read_epub(epub_path)
    chapters = list(book.get_items_of_type(ITEM_DOCUMENT))
    
    chapter_data = []
    for idx, item in enumerate(chapters, start=1):
        soup: BeautifulSoup = BeautifulSoup(item.get_content(), 'html.parser')
        chapter_text: str = soup.get_text(separator='\n', strip=True)
        words = chapter_text.split()
        
        if len(words) >= min_word_count:
            base_name = os.path.splitext(os.path.basename(item.get_name()))[0]
            chapter_data.append({
                'filename': f"{idx:03d}_{sanitize_filename(base_name)}",
                'text': ' '.join(words)
            })
    
    return chapter_data
