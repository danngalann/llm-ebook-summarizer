import os
import argparse
from llm import translate_text
from tqdm import tqdm

INPUT_DIR = "output"
TRANSLATIONS_DIR = "translations"

def process_text(chapter_filename: str, target_lang: str) -> None:
    # Read the content of the Markdown file
    with open(chapter_filename, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Translate the text using the provided function
    translated_text = translate_text(text, target_lang)
    
    # Create the translations output directory if it doesn't exist
    os.makedirs(TRANSLATIONS_DIR, exist_ok=True)
    
    # Prepare a new filename for the translated file in the translations directory
    base_name = os.path.basename(chapter_filename)
    name, ext = os.path.splitext(base_name)
    output_filename = os.path.join(TRANSLATIONS_DIR, f"{name}_{target_lang}{ext}")
    
    # Write the translated text to the new file
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(translated_text)
    
    print(f"Translated {chapter_filename} -> {output_filename}")

if __name__ == "__main__":
    # Set up argument parsing for target language
    parser = argparse.ArgumentParser(description="Translate all .md files in the output directory.")
    parser.add_argument("target_lang", type=str, help="Target language for translation (e.g., 'es' for Spanish)")
    args = parser.parse_args()
    
    # Gather all .md files in the input directory
    md_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".md")]
    
    # Process each file with a progress bar
    for md_file in tqdm(md_files, desc="Translating files"):
        process_text(md_file, args.target_lang)