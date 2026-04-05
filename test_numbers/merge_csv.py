import os
import csv
from pathlib import Path


def to_unsigned_32bit(value):
    if value < 0:
        return value & 0xFFFFFFFF
    return value


def merge_csv_files():
    """
    Merge all CSV files in ./inference_output into a single file.
    Skips comment lines and header lines.
    Sorts by the first column (Constant) in ascending order.
    Converts negative values exceeding 32 bits to unsigned representation.
    """
    input_dir   = Path("./inference_output")
    output_file = Path("./inference_output/merged.csv")

    if not input_dir.exists():
        print(f"Error: directory {input_dir} does not exist")
        return

    # Collect all CSV files, excluding merged.csv
    csv_files = sorted([f for f in input_dir.glob("*.csv")
                        if f.name != "merged.csv"])

    if not csv_files:
        print(f"Error: no CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV file(s)")
    print(f"Output file: {output_file}\n")

    all_data_rows    = []  # list of (constant_value, row_string)
    total_rows       = 0
    conversion_count = 0

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                row_count = 0
                for line in f:
                    line = line.strip()

                    if not line:
                        continue
                    if line.startswith('#'):
                        continue
                    if line.startswith('Constant,'):
                        continue

                    try:
                        parts          = line.split(',')
                        constant_value = int(parts[0])
                        original_value = constant_value
                        constant_unsigned = to_unsigned_32bit(constant_value)

                        if original_value != constant_unsigned:
                            conversion_count += 1
                            new_line = str(constant_unsigned) + ',' + ','.join(parts[1:])
                            all_data_rows.append((constant_unsigned, new_line))
                        else:
                            all_data_rows.append((constant_unsigned, line))

                        row_count += 1

                    except (ValueError, IndexError):
                        print(f"  WARNING: skipping invalid line: {line[:50]}...")
                        continue

            total_rows += row_count
            print(f"  Extracted {row_count} row(s)")

        except Exception as e:
            print(f"  ERROR: could not read {csv_file.name}: {e}")

    # Sort by Constant column
    print(f"\nSorting {total_rows} rows...")
    all_data_rows.sort(key=lambda x: x[0])

    # Write merged output
    print(f"Writing merged result to {output_file}...")

    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            f.write("Constant,prob_SPLUS,prob_SMINUS,prob_MINUSS,prob_BASE\n")
            for _, row in all_data_rows:
                f.write(row + "\n")

        print(f"\nMerge complete!")
        print(f"  Total rows merged:      {total_rows}")
        print(f"  Negative conversions:   {conversion_count}")
        print(f"  Sorted by:              Constant (ascending)")
        print(f"  Output:                 {output_file}")

        if all_data_rows:
            min_constant = all_data_rows[0][0]
            max_constant = all_data_rows[-1][0]
            print(f"  Constant range:         {min_constant} ~ {max_constant}")

            max_32bit = 0xFFFFFFFF
            if max_constant > max_32bit:
                print(f"  WARNING: values exceeding 32-bit range detected!")

    except Exception as e:
        print(f"\nERROR: failed to write output file: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("CSV Merge Tool (sort by Constant, negatives to unsigned)")
    print("=" * 60)
    print()

    merge_csv_files()