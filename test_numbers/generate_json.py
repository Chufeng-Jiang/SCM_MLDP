#!/usr/bin/env python3
"""
Batch generate JSON input files for all bit widths (17-32).
"""

import json
import os
import sys


def generate_all_json(input_folder="test_numbers", output_folder="test_inputs"):
    """
    Batch generate JSON input files for bit widths 17-32.
    """
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}\n")

    print("=" * 80)
    print("Batch generating JSON input files (17-32 bits)")
    print("=" * 80)

    success_count = 0
    fail_count    = 0

    for bit_number in range(17, 35):
        input_file  = os.path.join(input_folder,  f"{bit_number}bit_numbers.txt")
        output_file = os.path.join(output_folder, f"{bit_number}input.json")

        print(f"\nProcessing {bit_number} bits...")

        # Check input file
        if not os.path.exists(input_file):
            print(f"  SKIP: file not found - {input_file}")
            fail_count += 1
            continue

        try:
            # Read numbers
            numbers = []
            with open(input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        numbers.append(int(line))

            print(f"  Read: {len(numbers):,} numbers")

            # Build JSON data
            json_data = [
                {
                    "target":       number,
                    "current_mult": number,
                    "history":      []
                }
                for number in numbers
            ]

            # Save JSON
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=4)

            print(f"  Saved: {output_file}")
            success_count += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("Batch processing complete")
    print("=" * 80)
    print(f"  Success: {success_count} file(s)")
    if fail_count > 0:
        print(f"  Failed:  {fail_count} file(s)")
    print(f"  Output:  {output_folder}/")
    print("=" * 80)

    # List generated files
    if success_count > 0:
        print("\nGenerated files:")
        for bit_number in range(17, 33):
            output_file = os.path.join(output_folder, f"{bit_number}input.json")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"  {bit_number}input.json ({file_size:,} bytes)")


if __name__ == "__main__":
    input_folder  = sys.argv[1] if len(sys.argv) > 1 else "all_split_numbers"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "inference_input"

    generate_all_json(input_folder, output_folder)