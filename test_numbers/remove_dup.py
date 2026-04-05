
with open('./all_split_numbers/split_all.txt', 'r') as f:
    numbers = [int(line.strip()) for line in f if line.strip()]


unique_numbers = sorted(set(numbers) - {1})

print(unique_numbers)

with open('./all_split_numbers/allbit_numbers.txt', 'w') as f:
    for num in unique_numbers:
        f.write(f"{num}\n")