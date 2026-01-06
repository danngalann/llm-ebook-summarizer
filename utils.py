"""Utility functions for file and text processing."""

import re


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitizes a string to be used as a filename.
    
    Args:
        text (str): The text to sanitize.
        max_length (int): Maximum length of the sanitized filename.
    
    Returns:
        str: A sanitized filename-safe string.
    """
    # Remove or replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', text)
    # Replace multiple spaces with single underscore
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_.')
    return sanitized.lower()
