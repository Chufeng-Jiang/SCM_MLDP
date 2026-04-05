#!/usr/bin/env python3
"""
SCM data post-processing script
1. Rename files with negative names to 32-bit unsigned integers
2. Sort each file's content by the first number in ascending order
3. Save results to data_sorted folder, original files unchanged
"""

import os
import re
import sys


def get_output_path(data_path):
    parent = os.path.dirname(os.path.abspath(data_path))
    return os.path.join(parent, 'data_sorted')


def process_files(data_path, output_path):
    print("=== Step 1: Preparing output directory ===")
    os.makedirs(output_path, exist_ok=True)
    print(f"  Output directory: {output_path}\n")

    print("=== Step 2: Processing files (rename + sort) ===")

    def extract_first_number(line):
        m = re.search(r'\((\d+)', line)
        return int(m.group(1)) if m else float('inf')

    success_count = 0
    error_count = 0
    renamed_count = 0

    for filename in sorted(os.listdir(data_path)):
        src_path = os.path.join(data_path, filename)
        if not os.path.isfile(src_path):
            continue
        if not filename.lower().endswith('.txt'):
            continue

        # Handle negative file names -> 32-bit unsigned integer
        name, ext = os.path.splitext(filename)
        try:
            num = int(name)
            if num < 0:
                unsigned_num = num & 0xFFFFFFFF
                new_filename = f"{unsigned_num}{ext}"
                print(f"  Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            else:
                new_filename = filename
        except ValueError:
            new_filename = filename  # Non-numeric filename kept as-is

        dst_path = os.path.join(output_path, new_filename)

        try:
            with open(src_path, 'r') as f:
                lines = f.readlines()

            # All non-empty lines participate in sorting
            valid_lines = [l for l in lines if l.strip()]
            sorted_lines = sorted(valid_lines, key=extract_first_number)

            with open(dst_path, 'w') as f:
                for line in sorted_lines:
                    f.write(line if line.endswith('\n') else line + '\n')

            success_count += 1

        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            error_count += 1

    print(f"\n  Files renamed:  {renamed_count}")
    print(f"  Files success:  {success_count}")
    print(f"  Files failed:   {error_count}")


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else './data'

    if not os.path.exists(data_path):
        print(f"Error: path '{data_path}' does not exist")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) > 2 else get_output_path(data_path)

    print(f"Source:  {os.path.abspath(data_path)}")
    print(f"Output:  {os.path.abspath(output_path)}\n")

    process_files(data_path, output_path)
    print("\n=== Done ===")


if __name__ == '__main__':
    main()