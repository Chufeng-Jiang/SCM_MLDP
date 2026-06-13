def count_ones_in_binary(decimals_file):
    """
    Read a file containing decimal numbers and count the number of 1s
    in the binary representation of each number.

    Args:
        decimals_file: Path to the file containing decimal numbers.
    """
    try:
        # Open the input file in read mode
        with open(decimals_file, 'r') as file:
            line_number = 0

            # Print table header
            print("Decimal\tBinary\t\tNumber of 1s")
            print("-" * 30)

            # Process the file line by line
            for line in file:
                line_number += 1

                # Remove leading/trailing whitespace characters
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                try:
                    # Convert the current line to an integer
                    decimal_num = int(line)

                    # Convert the decimal number to a binary string
                    # bin() returns a string like '0b1010', so [2:] removes the '0b' prefix
                    binary_str = bin(decimal_num)[2:]

                    # Count the number of '1' bits in the binary string
                    ones_count = binary_str.count('1')

                    # Print the result in a tabular format
                    print(f"{decimal_num}\t{binary_str}\t\t{ones_count}")

                except ValueError:
                    # Handle lines that cannot be converted to valid integers
                    print(f"Line {line_number} is not a valid decimal number: '{line}'")

    except FileNotFoundError:
        # Handle the case where the input file does not exist
        print(f"Error: File '{decimals_file}' not found.")

    except Exception as e:
        # Handle any other unexpected errors
        print(f"An error occurred: {str(e)}")


def main():
    # Define the input file path
    input_file = "decimals.txt"

    # Process the input file and count 1s in binary representations
    count_ones_in_binary(input_file)


if __name__ == "__main__":
    main()