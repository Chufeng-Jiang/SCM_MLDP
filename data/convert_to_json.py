#!/usr/bin/env python3
"""
Convert data_sorted folder files to a single dpmink.json file.

File format:
    splus(mult, left_mult, right_mult, shift)   -> op=0
    sminus(mult, left_mult, right_mult, shift)  -> op=1
    minuss(mult, left_mult, right_mult, shift)  -> op=2

JSON structure:
    equations are stored in reverse order (largest mult first)
    left/right fields are 0-based indices into the equations array
    A fixed base entry (op=3, mult=1) is appended at the end
"""

import os
import re
import sys
import json


OP_MAP = {
    'splus':  0,
    'sminus': 1,
    'minuss': 2,
}


def parse_line(line):
    """Parse a single line like splus(2763851,1349,1099,11)
    Returns (op, mult, left_mult, right_mult, shift) or None if invalid.
    """
    line = line.strip()
    m = re.match(r'(splus|sminus|minuss)\((\d+),(\d+),(\d+),(\d+)\)', line)
    if not m:
        return None
    op_name    = m.group(1)
    mult       = int(m.group(2))
    left_mult  = int(m.group(3))
    right_mult = int(m.group(4))
    shift      = int(m.group(5))
    return OP_MAP[op_name], mult, left_mult, right_mult, shift


def build_equations(parsed_lines):
    """
    Build equations list in reverse order (largest mult first),
    with a fixed base entry (op=3, mult=1) appended at the end.

    left/right are 0-based indices into the equations array
    pointing to the entry whose mult matches left_mult/right_mult.
    """
    # Reverse: largest mult first
    reversed_lines = list(reversed(parsed_lines))

    # Build mult -> index map (0-based)
    # base entry is always last
    mult_to_index = {}
    for i, (op, mult, left_mult, right_mult, shift) in enumerate(reversed_lines):
        mult_to_index[mult] = i
    base_index = len(reversed_lines)
    mult_to_index[1] = base_index  # constant 1 always maps to base entry

    equations = []
    for op, mult, left_mult, right_mult, shift in reversed_lines:
        equations.append({
            "op":         op,
            "left":       mult_to_index[left_mult],
            "left_mult":  left_mult,
            "shift":      shift,
            "right":      mult_to_index[right_mult],
            "right_mult": right_mult,
            "mult":       mult,
        })

    # Fixed base entry
    equations.append({
        "op":         3,
        "left":       base_index,
        "left_mult":  1,
        "shift":      0,
        "right":      base_index,
        "right_mult": 1,
        "mult":       1,
    })

    return equations


def convert_file(file_path):
    """Convert a single file to a dict with c and equations."""
    with open(file_path, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]

    parsed_lines = []
    for line in lines:
        result = parse_line(line)
        if result is None:
            print(f"  WARNING: skipping unrecognized line: {line.strip()}")
            continue
        parsed_lines.append(result)

    if not parsed_lines:
        return None

    # c is the largest mult (last line after ascending sort)
    c = parsed_lines[-1][1]
    equations = build_equations(parsed_lines)

    return {"c": c, "equations": equations}


def main():
    data_path   = sys.argv[1] if len(sys.argv) > 1 else './data_sorted'
    output_file = sys.argv[2] if len(sys.argv) > 2 else './dpmink.json'

    if not os.path.exists(data_path):
        print(f"Error: path '{data_path}' does not exist")
        sys.exit(1)

    print(f"Source:  {os.path.abspath(data_path)}")
    print(f"Output:  {os.path.abspath(output_file)}\n")

    all_entries = []
    success = 0
    skipped = 0

    for filename in sorted(
        os.listdir(data_path),
        key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else float('inf')
    ):
        file_path = os.path.join(data_path, filename)
        if not os.path.isfile(file_path):
            continue
        if not filename.lower().endswith('.txt'):
            continue

        try:
            entry = convert_file(file_path)
            if entry is None:
                print(f"  SKIP (empty): {filename}")
                skipped += 1
                continue
            all_entries.append(entry)
            success += 1
        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            skipped += 1

    with open(output_file, 'w') as f:
        json.dump(all_entries, f, indent=2)

    print(f"Files processed: {success}")
    print(f"Files skipped:   {skipped}")
    print(f"Total entries:   {len(all_entries)}")
    print(f"\n=== Done -> {output_file} ===")


if __name__ == '__main__':
    main()