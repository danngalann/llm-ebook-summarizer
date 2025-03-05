import os
import argparse

def concat_md_files(input_folder: str, output_file: str = "book.md") -> None:
    """
    Reads all .md files in the given folder, concatenates them in alphabetical order,
    and writes the result to a single output file.

    Args:
        input_folder (str): Path to the folder containing Markdown files.
        output_file (str): Path to the output file. Defaults to "book.md".
    """
    # List all .md files in the folder
    md_files = [f for f in os.listdir(input_folder) if f.endswith(".md")]
    
    # Sort the files alphabetically
    md_files.sort()
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in md_files:
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
                # Add new lines as a separator between chapters
                outfile.write("\n\n")
    
    print(f"Concatenated {len(md_files)} files into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate all Markdown (.md) files in a folder into a single file (book.md)."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing the Markdown files."
    )
    args = parser.parse_args()

    concat_md_files(args.folder)
