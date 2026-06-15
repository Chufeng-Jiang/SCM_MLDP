#!/usr/bin/env python3
"""
Batch generate JSON input files for all bit widths from 17 to 32.

This script reads integer numbers from text files and converts them into
JSON input files for inference.

Expected input file format:

    all_split_numbers/17bit_numbers.txt
    all_split_numbers/18bit_numbers.txt
    ...
    all_split_numbers/32bit_numbers.txt

Each input text file should contain one integer per line.

Generated output file format:

    inference_input/17input.json
    inference_input/18input.json
    ...
    inference_input/32input.json

Each JSON file contains a list of objects. Each object has the format:

    {
        "target": number,
        "current_mult": number,
        "history": []
    }

where:
    - target: the original number to be processed
    - current_mult: initialized to the same value as target
    - history: an empty list used to store future operation history
"""

import json
import os
import sys


def generate_all_json(input_folder="all_split_numbers", output_folder="inference_input"):
    """
    Generate JSON input files for bit widths from 17 to 32.

    Parameters
    ----------
    input_folder : str
        Folder containing the input text files.
        Each file should be named like:

            17bit_numbers.txt
            18bit_numbers.txt
            ...
            32bit_numbers.txt

    output_folder : str
        Folder where the generated JSON files will be saved.
        Each output file will be named like:

            17input.json
            18input.json
            ...
            32input.json
    """

    # ------------------------------------------------------------
    # 1. Create the output folder if it does not already exist
    # ------------------------------------------------------------
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}\n")

    print("=" * 80)
    print("Batch generating JSON input files from 17 to 32 bits")
    print("=" * 80)

    # Counters for reporting how many files were successfully processed
    # and how many failed or were skipped.
    success_count = 0
    fail_count = 0

    # ------------------------------------------------------------
    # 2. Process each bit-width file
    # ------------------------------------------------------------
    # range(17, 33) means:
    #   17, 18, ..., 32
    for bit_number in range(17, 33):

        # Construct input and output file paths.
        input_file = os.path.join(
            input_folder,
            f"{bit_number}bit_numbers.txt"
        )

        output_file = os.path.join(
            output_folder,
            f"{bit_number}input.json"
        )

        print(f"\nProcessing {bit_number} bits...")

        # --------------------------------------------------------
        # 3. Check whether the input file exists
        # --------------------------------------------------------
        if not os.path.exists(input_file):
            print(f"  SKIP: file not found - {input_file}")
            fail_count += 1
            continue

        try:
            # ----------------------------------------------------
            # 4. Read all valid numbers from the input text file
            # ----------------------------------------------------
            numbers = []

            with open(input_file, "r") as f:
                for line in f:
                    line = line.strip()

                    # Skip empty lines
                    if line:
                        numbers.append(int(line))

            print(f"  Read: {len(numbers):,} numbers")

            # ----------------------------------------------------
            # 5. Convert numbers into JSON-compatible objects
            # ----------------------------------------------------
            # For each number, initialize:
            #
            #   target       = original number
            #   current_mult = current multiplication state
            #   history      = empty operation history
            #
            json_data = [
                {
                    "target": number,
                    "current_mult": number,
                    "history": []
                }
                for number in numbers
            ]

            # ----------------------------------------------------
            # 6. Save the generated data into a JSON file
            # ----------------------------------------------------
            with open(output_file, "w") as f:
                json.dump(json_data, f, indent=4)

            print(f"  Saved: {output_file}")
            success_count += 1

        except Exception as e:
            # Catch errors such as invalid integers, file permission issues,
            # or JSON writing errors.
            print(f"  ERROR: {e}")
            fail_count += 1

    # ------------------------------------------------------------
    # 7. Print final summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Batch processing complete")
    print("=" * 80)
    print(f"  Success: {success_count} file(s)")

    if fail_count > 0:
        print(f"  Failed:  {fail_count} file(s)")

    print(f"  Output:  {output_folder}/")
    print("=" * 80)

    # ------------------------------------------------------------
    # 8. List generated JSON files with file sizes
    # ------------------------------------------------------------
    if success_count > 0:
        print("\nGenerated files:")

        for bit_number in range(17, 33):
            output_file = os.path.join(
                output_folder,
                f"{bit_number}input.json"
            )

            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"  {bit_number}input.json ({file_size:,} bytes)")


if __name__ == "__main__":
    """
    Command-line usage:

        python generate_all_json.py

    This uses the default folders:

        input folder:  all_split_numbers
        output folder: inference_input

    You can also specify custom folders:

        python generate_all_json.py <input_folder> <output_folder>

    Example:

        python generate_all_json.py test_numbers test_inputs
    """

    # Use command-line arguments if provided.
    # Otherwise, use the default folders.
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "all_split_numbers"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "inference_input"

    generate_all_json(input_folder, output_folder)