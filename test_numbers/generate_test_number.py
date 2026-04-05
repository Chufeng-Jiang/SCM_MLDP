import os
import random
from datetime import datetime


def generate_random_binary_numbers_batch(folder_path, count_per_bit=1000, max_attempts_multiplier=10):
    """
    Generate a specified number of random odd binary numbers for each bit width (17-32 bits).
    Ensures generated decimal values do not already exist as filenames in data_sorted folder.
    If enough numbers cannot be generated within the attempt limit, generates as many as possible.

    Args:
        folder_path:              path to the data_sorted folder
        count_per_bit:            target number of values to generate per bit width
        max_attempts_multiplier:  multiplier for max attempts (actual max = count_per_bit * multiplier)

    Returns:
        dict: mapping each bit width to a sorted list of decimal values
    """

    # Step 1: Read all existing filenames (without extension) from data_sorted folder
    existing_numbers = set()

    if not os.path.exists(folder_path):
        print(f"WARNING: folder '{folder_path}' does not exist. "
              f"Only checking for duplicates among newly generated numbers.")
    else:
        for filename in os.listdir(folder_path):
            name_without_ext = os.path.splitext(filename)[0]
            if name_without_ext.isdigit():
                existing_numbers.add(int(name_without_ext))

        print(f"Found {len(existing_numbers):,} existing numbers in folder")

    print("=" * 80)
    print(f"Generating up to {count_per_bit:,} random odd numbers per bit width...")
    print("=" * 80)

    results          = {}
    generation_stats = {}

    for bits in range(17, 33):
        print(f"\nProcessing {bits}-bit numbers...")

        total_possible_odds = 2 ** (bits - 1)
        available_odds = total_possible_odds - sum(
            1 for n in existing_numbers
            if 2 ** (bits - 1) <= n < 2 ** bits and n % 2 == 1
        )

        print(f"  Available odd numbers: {available_odds:,} / {total_possible_odds:,}")

        target_count = min(count_per_bit, available_odds)
        if target_count < count_per_bit:
            print(f"  WARNING: target adjusted to {target_count:,} (limited availability)")

        generated_numbers      = set()
        attempts               = 0
        successful             = 0
        max_attempts           = count_per_bit * max_attempts_multiplier
        consecutive_failures   = 0
        max_consecutive_failures = count_per_bit

        while successful < target_count and attempts < max_attempts:
            attempts += 1

            # Build a random odd binary number with exactly `bits` bits:
            # - MSB fixed to 1 (ensures exactly bits digits)
            # - LSB fixed to 1 (ensures odd)
            binary_str = '1'
            for _ in range(bits - 2):
                binary_str += str(random.randint(0, 1))
            binary_str += '1'

            decimal_value = int(binary_str, 2)

            if decimal_value not in existing_numbers and decimal_value not in generated_numbers:
                generated_numbers.add(decimal_value)
                successful += 1
                consecutive_failures = 0

                if successful % 1000 == 0:
                    progress   = successful / target_count * 100
                    bar_length = 40
                    filled     = int(bar_length * successful / target_count)
                    bar        = '#' * filled + '.' * (bar_length - filled)
                    print(f"  Progress: [{bar}] {progress:.1f}% ({successful:,}/{target_count:,})", end='\r')
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"\n  WARNING: {consecutive_failures:,} consecutive failures — "
                          f"pool may be exhausted")
                    print(f"  Generated so far: {successful:,} (target: {target_count:,})")
                    break

        results[bits] = sorted(list(generated_numbers))

        generation_stats[bits] = {
            'target':              target_count,
            'actual':              len(results[bits]),
            'attempts':            attempts,
            'success_rate':        len(results[bits]) / attempts if attempts > 0 else 0,
            'available_odds':      available_odds,
            'total_possible_odds': total_possible_odds,
        }

        if len(results[bits]) >= target_count:
            print(f"\n  Done! Generated {len(results[bits]):,} unique {bits}-bit odd numbers "
                  f"({attempts:,} attempts)")
        else:
            print(f"\n  Partial! Generated {len(results[bits]):,}/{target_count:,} "
                  f"({attempts:,} attempts)")
            print(f"  Success rate: {generation_stats[bits]['success_rate'] * 100:.2f}%")

        existing_numbers.update(generated_numbers)

    # Overall statistics table
    print("\n" + "=" * 80)
    print("Generation Statistics")
    print("=" * 80)
    print(f"{'Bits':<6} {'Target':<10} {'Actual':<10} {'Completion':<12} {'Attempts':<12} {'Success Rate'}")
    print("-" * 80)

    for bits in sorted(generation_stats.keys()):
        s               = generation_stats[bits]
        completion_rate = s['actual'] / s['target'] * 100 if s['target'] > 0 else 0
        print(f"{bits:<6} {s['target']:<10,} {s['actual']:<10,} "
              f"{completion_rate:<11.1f}% {s['attempts']:<12,} {s['success_rate'] * 100:>6.2f}%")

    print("=" * 80)

    return results


def save_results_to_files(results, output_folder="test_numbers"):
    """
    Save each bit width's results to a separate text file.

    Args:
        results:       dict returned by generate_random_binary_numbers_batch
        output_folder: path to the output folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nCreated output folder: {output_folder}")

    print("\n" + "=" * 80)
    print("Saving results to files...")
    print("=" * 80)

    for bits in sorted(results.keys()):
        numbers  = results[bits]
        filename = os.path.join(output_folder, f"{bits}bit_numbers.txt")

        with open(filename, 'w', encoding='utf-8') as f:
            for number in numbers:
                f.write(f"{number}\n")

        print(f"  {bits} bits: saved {len(numbers):,} numbers -> {filename}")

    print("=" * 80)


def save_summary(results, output_folder="test_numbers", target_count=10000):
    """
    Save a statistics summary to a text file.

    Args:
        results:       dict of generated numbers per bit width
        output_folder: path to the output folder
        target_count:  target number of values per bit width
    """
    summary_file = os.path.join(output_folder, "summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Generated 17-32 bit odd binary numbers — statistics summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target count: {target_count:,} per bit width\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Bits':<6} {'Count':<10} {'Completion':<12} {'Min':<15} {'Max':<15} {'Filename'}\n")
        f.write("-" * 80 + "\n")

        total_numbers = 0
        total_target  = 0

        for bits in sorted(results.keys()):
            numbers         = results[bits]
            total_numbers  += len(numbers)
            total_target   += target_count
            min_val         = min(numbers) if numbers else 0
            max_val         = max(numbers) if numbers else 0
            filename        = f"{bits}bit_numbers.txt"
            completion_rate = len(numbers) / target_count * 100 if target_count > 0 else 0

            f.write(f"{bits:<6} {len(numbers):<10,} {completion_rate:<11.1f}% "
                    f"{min_val:<15,} {max_val:<15,} {filename}\n")

        f.write("-" * 80 + "\n")
        f.write(f"Total: {total_numbers:,} / {total_target:,} "
                f"({total_numbers / total_target * 100:.1f}% complete)\n")
        f.write(f"Bit range: 17-32 bits (16 widths)\n")

        # Theoretical ranges
        f.write("\n" + "=" * 80 + "\n")
        f.write("Theoretical value ranges\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Bits':<6} {'Min':<20} {'Max':<20} {'Total odd numbers'}\n")
        f.write("-" * 80 + "\n")

        for bits in sorted(results.keys()):
            theoretical_min = 2 ** (bits - 1)
            theoretical_max = (2 ** bits) - 1
            total_odds      = 2 ** (bits - 1)
            f.write(f"{bits:<6} {theoretical_min:<20,} {theoretical_max:<20,} {total_odds:,}\n")

        # Incomplete bit widths
        incomplete = [bits for bits in results.keys() if len(results[bits]) < target_count]
        if incomplete:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Bit widths that did not reach target\n")
            f.write("=" * 80 + "\n")
            for bits in incomplete:
                actual     = len(results[bits])
                completion = actual / target_count * 100
                f.write(f"{bits} bits: {actual:,} / {target_count:,} ({completion:.1f}%)\n")

    print(f"\nSummary saved to {summary_file}")


def display_summary(results, target_count=10000):
    """
    Print a formatted summary of generation results.

    Args:
        results:      dict of generated numbers per bit width
        target_count: target number of values per bit width
    """
    print("\n" + "=" * 80)
    print("Generation Results Summary")
    print("=" * 80)
    print(f"{'Bits':<6} {'Count':<10} {'Completion':<12} {'Min':<15} {'Max':<15}")
    print("-" * 80)

    total_numbers = 0
    total_target  = 0

    for bits in sorted(results.keys()):
        numbers         = results[bits]
        total_numbers  += len(numbers)
        total_target   += target_count
        min_val         = min(numbers) if numbers else 0
        max_val         = max(numbers) if numbers else 0
        completion_rate = len(numbers) / target_count * 100 if target_count > 0 else 0

        if completion_rate >= 100:
            status = "OK "
        elif completion_rate >= 90:
            status = "~  "
        else:
            status = "!  "

        print(f"{status} {bits:<4} {len(numbers):<10,} {completion_rate:<11.1f}% "
              f"{min_val:<15,} {max_val:<15,}")

    print("-" * 80)
    overall_completion = total_numbers / total_target * 100 if total_target > 0 else 0
    print(f"Total: {total_numbers:,} / {total_target:,} ({overall_completion:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    data_sorted_folder      = "../data/data_sorted"
    output_folder           = "all_numbers"
    count_per_bit           = 1000
    max_attempts_multiplier = 10

    print("\n" + "=" * 80)
    print("Batch Random Odd Binary Number Generator")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Existing data folder:  {data_sorted_folder}")
    print(f"  Output folder:         {output_folder}")
    print(f"  Target per bit width:  {count_per_bit:,}")
    print(f"  Bit width range:       17-32 (16 widths)")
    print(f"  Expected total:        {count_per_bit * 16:,}")
    print(f"  Max attempts per bit:  {count_per_bit * max_attempts_multiplier:,}")
    print(f"  Start time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nNote: if available numbers are insufficient for a bit width, "
          "all available numbers will be generated.")
    print("=" * 80)

    start_time = datetime.now()

    results = generate_random_binary_numbers_batch(
        data_sorted_folder,
        count_per_bit,
        max_attempts_multiplier
    )

    if results:
        display_summary(results, count_per_bit)
        save_results_to_files(results, output_folder)
        save_summary(results, output_folder, count_per_bit)

        end_time          = datetime.now()
        elapsed           = end_time - start_time
        fully_completed   = sum(1 for nums in results.values() if len(nums) >= count_per_bit)
        partially_completed = len(results) - fully_completed

        print("\n" + "=" * 80)
        print("All operations complete!")
        print("=" * 80)
        print(f"  Elapsed time:      {elapsed}")
        print(f"  Output folder:     {output_folder}/")
        print(f"  Generated files:   17bit_numbers.txt ~ 32bit_numbers.txt, summary.txt")
        print(f"\n  Completion:")
        print(f"    Fully complete:  {fully_completed} bit width(s)")
        if partially_completed > 0:
            print(f"    Partial:         {partially_completed} bit width(s) (see summary.txt)")

        total_generated = sum(len(nums) for nums in results.values())
        total_target    = count_per_bit * 16
        print(f"    Total generated: {total_generated:,} / {total_target:,} "
              f"({total_generated / total_target * 100:.1f}%)")
        print("=" * 80)

    else:
        print("ERROR: no numbers were generated")