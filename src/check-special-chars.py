#!/usr/bin/env python3
"""
CSV UTF-8 and Windows Special Character Checker

This script checks a CSV file for:
1. Valid UTF-8 encoding
2. Microsoft Windows special characters (CP1252)

Usage: python check_csv.py <file_path>
Example: python check_csv.py ../data/data.csv
"""

import sys
import os
import csv
from typing import List, Tuple, Dict

# Microsoft Windows CP1252 special characters (bytes 128-159)
# These are often problematic when converting between encodings
WINDOWS_SPECIAL_CHARS = {
    0x80: '€',  # Euro sign
    0x82: '‚',  # Single low-9 quotation mark
    0x83: 'ƒ',  # Latin small letter f with hook
    0x84: '„',  # Double low-9 quotation mark
    0x85: '…',  # Horizontal ellipsis
    0x86: '†',  # Dagger
    0x87: '‡',  # Double dagger
    0x88: 'ˆ',  # Modifier letter circumflex accent
    0x89: '‰',  # Per mille sign
    0x8A: 'Š',  # Latin capital letter S with caron
    0x8B: '‹',  # Single left-pointing angle quotation mark
    0x8C: 'Œ',  # Latin capital ligature OE
    0x8E: 'Ž',  # Latin capital letter Z with caron
    0x91: ''',  # Left single quotation mark
    0x92: ''',  # Right single quotation mark
    0x93: '"',  # Left double quotation mark
    0x94: '"',  # Right double quotation mark
    0x95: '•',  # Bullet
    0x96: '–',  # En dash
    0x97: '—',  # Em dash
    0x98: '˜',  # Small tilde
    0x99: '™',  # Trade mark sign
    0x9A: 'š',  # Latin small letter s with caron
    0x9B: '›',  # Single right-pointing angle quotation mark
    0x9C: 'œ',  # Latin small ligature oe
    0x9E: 'ž',  # Latin small letter z with caron
    0x9F: 'Ÿ',  # Latin capital letter Y with diaeresis
}

def check_utf8_validity(file_path: str) -> Tuple[bool, List[str]]:
    """
    Check if file contains valid UTF-8 characters.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_num = 1
            for line in file:
                try:
                    # Try to encode and decode to catch any issues
                    line.encode('utf-8').decode('utf-8')
                except UnicodeError as e:
                    errors.append(f"Line {line_num}: {str(e)}")
                line_num += 1
    except UnicodeDecodeError as e:
        errors.append(f"File encoding error: {str(e)}")
        return False, errors
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
        return False, errors
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
        return False, errors
    
    return len(errors) == 0, errors

def find_windows_special_chars(file_path: str) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Find Microsoft Windows special characters in the file.
    
    Returns:
        Dictionary with character as key and list of (row, col, context) as value
    """
    found_chars = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            reader = csv.reader(file)
            row_num = 1
            
            for row in reader:
                col_num = 1
                for cell in row:
                    for char_pos, char in enumerate(cell):
                        char_code = ord(char)
                        if char_code in WINDOWS_SPECIAL_CHARS:
                            char_name = WINDOWS_SPECIAL_CHARS[char_code]
                            if char not in found_chars:
                                found_chars[char] = []
                            
                            # Get context (10 chars before and after)
                            start = max(0, char_pos - 10)
                            end = min(len(cell), char_pos + 11)
                            context = cell[start:end].replace('\n', '\\n').replace('\r', '\\r')
                            
                            found_chars[char].append((row_num, col_num, context))
                    col_num += 1
                row_num += 1
                
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return found_chars

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_csv.py <file_path>")
        print("Example: python check_csv.py ../data/data.csv")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Convert to absolute path for better error messages
    abs_path = os.path.abspath(file_path)
    
    print(f"Checking file: {file_path}")
    print(f"Absolute path: {abs_path}")
    print("-" * 60)
    
    # Check UTF-8 validity
    print("1. Checking UTF-8 validity...")
    is_valid_utf8, utf8_errors = check_utf8_validity(file_path)
    
    if is_valid_utf8:
        print("✓ File contains valid UTF-8 characters")
    else:
        print("✗ UTF-8 validation errors found:")
        for error in utf8_errors:
            print(f"  - {error}")
    
    print()
    
    # Check for Windows special characters
    print("2. Checking for Microsoft Windows special characters...")
    windows_chars = find_windows_special_chars(file_path)
    
    if not windows_chars:
        print("✓ No Windows special characters found")
    else:
        print("✗ Windows special characters found:")
        for char, locations in windows_chars.items():
            char_code = ord(char)
            char_name = WINDOWS_SPECIAL_CHARS.get(char_code, "Unknown")
            print(f"\n  Character: '{char}' (U+{char_code:04X}, {char_name})")
            print(f"  Found {len(locations)} time(s):")
            
            for row, col, context in locations[:5]:  # Show first 5 occurrences
                print(f"    Row {row}, Column {col}: ...{context}...")
            
            if len(locations) > 5:
                print(f"    ... and {len(locations) - 5} more occurrences")
    
    print()
    print("-" * 60)
    
    # Summary
    if is_valid_utf8 and not windows_chars:
        print("✓ File passed all checks!")
        sys.exit(0)
    else:
        print("✗ File has issues that need attention")
        sys.exit(1)

if __name__ == "__main__":
    main()