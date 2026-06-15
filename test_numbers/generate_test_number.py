#!/usr/bin/env python3
"""
Batch Random Odd Binary Number Generator

This script generates random odd integers for each bit width from 17 to 32.

For each bit width, the script:
    1. Generates random binary numbers with exactly that many bits.
    2. Forces the most significant bit (MSB) to 1.
       - This guarantees the number has the required bit width.
    3. Forces the least significant bit (LSB) to 1.
       - This guarantees the number is odd.
    4. Converts the binary number to decimal.
    5. Checks that the number does not already exist in the data_sorted folder.
    6. Saves the generated numbers into separate text files.

Example output files:

    all_numbers/17bit_numbers.txt
    all_numbers/18bit_numbers.txt
    ...
    all_numbers/32bit_numbers.txt

Each file contains one generated decimal number per line.

The script also creates:

    all_numbers/summary.txt

which records generation statistics, theoretical ranges, and completion status.
"""

import os
import random
from datetime import datetime


def generate_random_binary_numbers_batch(
    folder_path,
    count_per_bit=1000,
    max_attempts_multiplier=10
):
    """
    Generate random unique odd integers for each bit width from 17 to 32.

    The generated numbers are checked against existing numbers in the given
    folder. Existing numbers are read from filenames, assuming the filenames
    are decimal values such as:

        123456.txt
        987654.pi

    Only filenames whose base name is numeric are considered.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing existing data files.
        The script uses this folder to avoid generating duplicate numbers.

    count_per_bit : int, optional
        Target number of random values to generate for each bit width.
        Default is 1000.

    max_attempts_multiplier : int, optional
        Maximum attempt multiplier.
        The maximum number of random-generation attempts per bit width is:

            count_per_bit * max_attempts_multiplier

        This prevents infinite loops when the available number pool is small.

    Returns
    -------
    dict
        A dictionary mapping each bit width to a sorted list of generated
        decimal values.

        Example:

            {
                17: [65537, 65539, ...],
                18: [131073, 131077, ...],
                ...
            }
    """

    # ------------------------------------------------------------
    # 1. Load existing numbers from the data_sorted folder
    # ------------------------------------------------------------
    existing_numbers = set()

    if not os.path.exists(folder_path):
        print(
            f"WARNING: folder '{folder_path}' does not exist. "
            f"Only checking for duplicates among newly generated numbers."
        )
    else:
        for filename in os.listdir(folder_path):

            # Remove file extension.
            # Example:
            #   "12345.pi" -> "12345"
            name_without_ext = os.path.splitext(filename)[0]

            # Only treat numeric filenames as existing numbers.
            if name_without_ext.isdigit():
                existing_numbers.add(int(name_without_ext))

        print(f"Found {len(existing_numbers):,} existing numbers in folder")

    print("=" * 80)
    print(f"Generating up to {count_per_bit:,} random odd numbers per bit width...")
    print("=" * 80)

    results = {}
    generation_stats = {}

    # ------------------------------------------------------------
    # 2. Generate numbers for each bit width from 17 to 32
    # ------------------------------------------------------------
    for bits in range(17, 33):
        print(f"\nProcessing {bits}-bit numbers...")

        # Total odd numbers with exactly `bits` bits:
        #
        #   Range of bits-bit numbers:
        #       [2^(bits-1), 2^bits - 1]
        #
        #   Half of them are odd, so the total number of odd values is:
        #       2^(bits-1)
        #
        total_possible_odds = 2 ** (bits - 1)

        # Count how many valid odd numbers in this bit range are already used.
        existing_odds_for_this_bit = sum(
            1
            for n in existing_numbers
            if 2 ** (bits - 1) <= n < 2 ** bits and n % 2 == 1
        )

        available_odds = total_possible_odds - existing_odds_for_this_bit

        print(
            f"  Available odd numbers: "
            f"{available_odds:,} / {total_possible_odds:,}"
        )

        # If fewer numbers are available than requested, lower the target.
        target_count = min(count_per_bit, available_odds)

        if target_count < count_per_bit:
            print(
                f"  WARNING: target adjusted to {target_count:,} "
                f"(limited availability)"
            )

        generated_numbers = set()
        attempts = 0
        successful = 0

        # Maximum number of attempts for this bit width.
        max_attempts = count_per_bit * max_attempts_multiplier

        # If too many attempts fail consecutively, the candidate pool may
        # be nearly exhausted, so we stop early.
        consecutive_failures = 0
        max_consecutive_failures = count_per_bit

        # --------------------------------------------------------
        # 3. Random generation loop
        # --------------------------------------------------------
        while successful < target_count and attempts < max_attempts:
            attempts += 1

            # Build a random binary string with exactly `bits` bits.
            #
            # Example for bits = 6:
            #
            #   1 + random middle bits + 1
            #
            # Possible result:
            #
            #   101101
            #
            # The first 1 ensures this is exactly a 6-bit number.
            # The last 1 ensures the decimal value is odd.
            binary_str = "1"

            for _ in range(bits - 2):
                binary_str += str(random.randint(0, 1))

            binary_str += "1"

            # Convert the binary string to a decimal integer.
            decimal_value = int(binary_str, 2)

            # Keep the number only if:
            #   1. It does not already exist in the data folder.
            #   2. It was not already generated in this run.
            if (
                decimal_value not in existing_numbers
                and decimal_value not in generated_numbers
            ):
                generated_numbers.add(decimal_value)
                successful += 1
                consecutive_failures = 0

                # Print progress every 1000 successful generations.
                if successful % 1000 == 0:
                    progress = successful / target_count * 100
                    bar_length = 40
                    filled = int(bar_length * successful / target_count)
                    bar = "#" * filled + "." * (bar_length - filled)

                    print(
                        f"  Progress: [{bar}] {progress:.1f}% "
                        f"({successful:,}/{target_count:,})",
                        end="\r"
                    )

            else:
                consecutive_failures += 1

                # Stop early if too many duplicate attempts occur in a row.
                if consecutive_failures >= max_consecutive_failures:
                    print(
                        f"\n  WARNING: {consecutive_failures:,} consecutive "
                        f"failures — pool may be exhausted"
                    )
                    print(
                        f"  Generated so far: {successful:,} "
                        f"(target: {target_count:,})"
                    )
                    break

        # Store sorted results for this bit width.
        results[bits] = sorted(list(generated_numbers))

        # Record generation statistics.
        generation_stats[bits] = {
            "target": target_count,
            "actual": len(results[bits]),
            "attempts": attempts,
            "success_rate": len(results[bits]) / attempts if attempts > 0 else 0,
            "available_odds": available_odds,
            "total_possible_odds": total_possible_odds,
        }

        if len(results[bits]) >= target_count:
            print(
                f"\n  Done! Generated {len(results[bits]):,} unique "
                f"{bits}-bit odd numbers ({attempts:,} attempts)"
            )
        else:
            print(
                f"\n  Partial! Generated {len(results[bits]):,}/"
                f"{target_count:,} ({attempts:,} attempts)"
            )
            print(
                f"  Success rate: "
                f"{generation_stats[bits]['success_rate'] * 100:.2f}%"
            )

        # Add newly generated numbers to the global existing set so that
        # later bit widths cannot accidentally reuse them.
        existing_numbers.update(generated_numbers)

    # ------------------------------------------------------------
    # 4. Print overall generation statistics
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(
        f"{'Bits':<6} {'Target':<10} {'Actual':<10} "
        f"{'Completion':<12} {'Attempts':<12} {'Success Rate'}"
    )
    print("-" * 80)

    for bits in sorted(generation_stats.keys()):
        s = generation_stats[bits]
        completion_rate = s["actual"] / s["target"] * 100 if s["target"] > 0 else 0

        print(
            f"{bits:<6} {s['target']:<10,} {s['actual']:<10,} "
            f"{completion_rate:<11.1f}% {s['attempts']:<12,} "
            f"{s['success_rate'] * 100:>6.2f}%"
        )

    print("=" * 80)

    return results


def save_results_to_files(results, output_folder="test_numbers"):
    """
    Save generated numbers to separate text files.

    Parameters
    ----------
    results : dict
        Dictionary returned by generate_random_binary_numbers_batch.

    output_folder : str, optional
        Folder where output files will be saved.
        Default is "test_numbers".
    """

    # Create output folder if needed.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nCreated output folder: {output_folder}")

    print("\n" + "=" * 80)
    print("Saving results to files...")
    print("=" * 80)

    for bits in sorted(results.keys()):
        numbers = results[bits]
        filename = os.path.join(output_folder, f"{bits}bit_numbers.txt")

        # Save one number per line.
        with open(filename, "w", encoding="utf-8") as f:
            for number in numbers:
                f.write(f"{number}\n")

        print(f"  {bits} bits: saved {len(numbers):,} numbers -> {filename}")

    print("=" * 80)


def save_summary(results, output_folder="test_numbers", target_count=10000):
    """
    Save a detailed generation summary to summary.txt.

    The summary includes:
        - generation timestamp
        - target count
        - actual count per bit width
        - completion rate
        - minimum and maximum generated values
        - theoretical value ranges
        - incomplete bit widths, if any

    Parameters
    ----------
    results : dict
        Dictionary of generated numbers per bit width.

    output_folder : str, optional
        Folder where summary.txt will be saved.

    target_count : int, optional
        Target number of values per bit width.
    """

    summary_file = os.path.join(output_folder, "summary.txt")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Generated 17-32 bit odd binary numbers — statistics summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target count: {target_count:,} per bit width\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"{'Bits':<6} {'Count':<10} {'Completion':<12} "
            f"{'Min':<15} {'Max':<15} {'Filename'}\n"
        )
        f.write("-" * 80 + "\n")

        total_numbers = 0
        total_target = 0

        for bits in sorted(results.keys()):
            numbers = results[bits]
            total_numbers += len(numbers)
            total_target += target_count

            min_val = min(numbers) if numbers else 0
            max_val = max(numbers) if numbers else 0
            filename = f"{bits}bit_numbers.txt"
            completion_rate = len(numbers) / target_count * 100 if target_count > 0 else 0

            f.write(
                f"{bits:<6} {len(numbers):<10,} {completion_rate:<11.1f}% "
                f"{min_val:<15,} {max_val:<15,} {filename}\n"
            )

        f.write("-" * 80 + "\n")
        f.write(
            f"Total: {total_numbers:,} / {total_target:,} "
            f"({total_numbers / total_target * 100:.1f}% complete)\n"
        )
        f.write("Bit range: 17-32 bits (16 widths)\n")

        # --------------------------------------------------------
        # Write theoretical ranges for each bit width
        # --------------------------------------------------------
        f.write("\n" + "=" * 80 + "\n")
        f.write("Theoretical value ranges\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Bits':<6} {'Min':<20} {'Max':<20} {'Total odd numbers'}\n")
        f.write("-" * 80 + "\n")

        for bits in sorted(results.keys()):
            theoretical_min = 2 ** (bits - 1)
            theoretical_max = (2 ** bits) - 1
            total_odds = 2 ** (bits - 1)

            f.write(
                f"{bits:<6} {theoretical_min:<20,} "
                f"{theoretical_max:<20,} {total_odds:,}\n"
            )

        # Record bit widths that did not reach the target.
        incomplete = [
            bits
            for bits in results.keys()
            if len(results[bits]) < target_count
        ]

        if incomplete:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Bit widths that did not reach target\n")
            f.write("=" * 80 + "\n")

            for bits in incomplete:
                actual = len(results[bits])
                completion = actual / target_count * 100

                f.write(
                    f"{bits} bits: {actual:,} / {target_count:,} "
                    f"({completion:.1f}%)\n"
                )

    print(f"\nSummary saved to {summary_file}")


def display_summary(results, target_count=10000):
    """
    Print a formatted generation summary to the terminal.

    Parameters
    ----------
    results : dict
        Dictionary of generated numbers per bit width.

    target_count : int, optional
        Target number of values per bit width.
    """

    print("\n" + "=" * 80)
    print("Generation Results Summary")
    print("=" * 80)
    print(f"{'Bits':<6} {'Count':<10} {'Completion':<12} {'Min':<15} {'Max':<15}")
    print("-" * 80)

    total_numbers = 0
    total_target = 0

    for bits in sorted(results.keys()):
        numbers = results[bits]
        total_numbers += len(numbers)
        total_target += target_count

        min_val = min(numbers) if numbers else 0
        max_val = max(numbers) if numbers else 0
        completion_rate = len(numbers) / target_count * 100 if target_count > 0 else 0

        # Status label:
        #   OK = fully complete
        #   ~  = mostly complete
        #   !  = significantly incomplete
        if completion_rate >= 100:
            status = "OK "
        elif completion_rate >= 90:
            status = "~  "
        else:
            status = "!  "

        print(
            f"{status} {bits:<4} {len(numbers):<10,} "
            f"{completion_rate:<11.1f}% "
            f"{min_val:<15,} {max_val:<15,}"
        )

    print("-" * 80)

    overall_completion = total_numbers / total_target * 100 if total_target > 0 else 0

    print(f"Total: {total_numbers:,} / {total_target:,} ({overall_completion:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    """
    Main execution configuration.

    Default configuration:
        - Existing data folder:
              ../data/data_sorted

        - Output folder:
              all_numbers

        - Target generated numbers per bit width:
              1000

        - Bit-width range:
              17 to 32

    Running the script:

        python generate_numbers.py
    """

    data_sorted_folder = "../data/data_sorted"
    output_folder = "all_numbers"
    count_per_bit = 1000
    max_attempts_multiplier = 10

    print("\n" + "=" * 80)
    print("Batch Random Odd Binary Number Generator")
    print("=" * 80)
    print("Configuration:")
    print(f"  Existing data folder:  {data_sorted_folder}")
    print(f"  Output folder:         {output_folder}")
    print(f"  Target per bit width:  {count_per_bit:,}")
    print("  Bit width range:       17-32 (16 widths)")
    print(f"  Expected total:        {count_per_bit * 16:,}")
    print(f"  Max attempts per bit:  {count_per_bit * max_attempts_multiplier:,}")
    print(f"  Start time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(
        "\nNote: if available numbers are insufficient for a bit width, "
        "all available numbers will be generated."
    )
    print("=" * 80)

    start_time = datetime.now()

    # Generate random odd numbers.
    results = generate_random_binary_numbers_batch(
        data_sorted_folder,
        count_per_bit,
        max_attempts_multiplier
    )

    if results:
        # Print summary to terminal.
        display_summary(results, count_per_bit)

        # Save generated numbers to text files.
        save_results_to_files(results, output_folder)

        # Save a detailed summary report.
        save_summary(results, output_folder, count_per_bit)

        end_time = datetime.now()
        elapsed = end_time - start_time

        fully_completed = sum(
            1
            for nums in results.values()
            if len(nums) >= count_per_bit
        )

        partially_completed = len(results) - fully_completed

        print("\n" + "=" * 80)
        print("All operations complete!")
        print("=" * 80)
        print(f"  Elapsed time:      {elapsed}")
        print(f"  Output folder:     {output_folder}/")
        print("  Generated files:   17bit_numbers.txt ~ 32bit_numbers.txt, summary.txt")
        print("\n  Completion:")
        print(f"    Fully complete:  {fully_completed} bit width(s)")

        if partially_completed > 0:
            print(
                f"    Partial:         {partially_completed} bit width(s) "
                f"(see summary.txt)"
            )

        total_generated = sum(len(nums) for nums in results.values())
        total_target = count_per_bit * 16

        print(
            f"    Total generated: {total_generated:,} / {total_target:,} "
            f"({total_generated / total_target * 100:.1f}%)"
        )
        print("=" * 80)

    else:
        print("ERROR: no numbers were generated")