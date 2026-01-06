import os
import argparse
from llm import translate_text_batch
from tqdm import tqdm

INPUT_DIR = "output"
TRANSLATIONS_DIR = "translations"

def translate_all_files(target_lang: str) -> None:
    """
    Translates all markdown files in batch using vLLM.
    
    Args:
        target_lang (str): Target language for translation.
    """
    # Gather all .md files in the input directory
    md_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".md")]
    
    if not md_files:
        print("No markdown files found to translate.")
        return
    
    # Create the translations output directory if it doesn't exist
    os.makedirs(TRANSLATIONS_DIR, exist_ok=True)
    
    print(f"Loading {len(md_files)} files...")
    
    # Read all files
    texts = []
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as file:
            texts.append(file.read())
    
    print(f"Translating {len(texts)} files in batch...")
    
    # Translate all files in one batch
    translated_texts = translate_text_batch(texts, target_lang)
    
    # Save results with progress bar
    for md_file, translated_text in tqdm(zip(md_files, translated_texts), total=len(md_files), desc="Saving translations"):
        base_name = os.path.basename(md_file)
        name, ext = os.path.splitext(base_name)
        output_filename = os.path.join(TRANSLATIONS_DIR, f"{name}_{target_lang}{ext}")
        
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(translated_text)
        
        print(f"Translated {md_file} -> {output_filename}")

if __name__ == "__main__":
    # Set up argument parsing for target language
    parser = argparse.ArgumentParser(description="Translate all .md files in the output directory.")
    parser.add_argument("target_lang", type=str, help="Target language for translation (e.g., 'es' for Spanish)")
    args = parser.parse_args()
    
    translate_all_files(args.target_lang)
    
    print("Translation complete. All files have been translated.")