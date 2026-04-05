# fixed_data_split.py
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os

RANDOM_SEED = 42

def set_random_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def get_bit_range(c_value):
    if c_value == 0:
        return 0
    bits = c_value.bit_length()
    return bits

def custom_data_split(json_path, test_ratio=0.1, random_state=RANDOM_SEED):
    """
    Custom data split strategy (split by C value):
    1. Find samples containing op=2, randomly select 10000 for training set
    2. Remaining samples are grouped by bit length of their C value,
       within each bit group, split by (1-test_ratio) ratio into train/test
    3. Each sample = one complete JSON object {"c": ..., "equations": [...]}
    4. Each C value should be unique
    """
    print("="*60)
    print("Custom Data Split Strategy")
    print("="*60)

    set_random_seed(random_state)

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    print(f"Total samples in dataset: {len(raw_data)}")

    # Check uniqueness of C values
    c_values_seen = set()
    duplicate_c_values = []
    for sample in raw_data:
        c_value = sample["c"]
        if c_value in c_values_seen:
            duplicate_c_values.append(c_value)
        c_values_seen.add(c_value)

    if duplicate_c_values:
        print(f"WARNING: Found {len(duplicate_c_values)} duplicate C values!")
        print(f"   First 10: {duplicate_c_values[:10]}")
        print(f"   This suggests the data has multiple decomposition methods for the same C value.")
    else:
        print(f"All C values are unique ({len(c_values_seen)} unique C values)")

    # Categorize samples
    op2_samples = []
    other_samples = []

    for sample in raw_data:
        equations = sample["equations"]

        # Check if sample contains op=2
        has_op2 = any(eq.get("op", 0) == 2 for eq in equations)

        if has_op2:
            op2_samples.append(sample)
        else:
            other_samples.append(sample)

    print(f"\nSample categories:")
    print(f"  Samples with OP=2: {len(op2_samples)}")
    print(f"  Other samples: {len(other_samples)}")

    # 1. Handle OP=2 samples: randomly select 10000 for training set
    print(f"\n1. Processing samples with OP=2: {len(op2_samples)}")
    rng = np.random.RandomState(random_state)

    if len(op2_samples) > 10000:
        # Randomly select 10000 for training set
        indices = rng.permutation(len(op2_samples))
        op2_train = [op2_samples[i] for i in indices[:10000]]
        op2_test  = [op2_samples[i] for i in indices[10000:]]
    else:
        # If fewer than 10000, all go to training set
        op2_train = op2_samples
        op2_test  = []

    print(f"   OP=2 samples: {len(op2_train)} train, {len(op2_test)} test")

    # 2. Split other samples by bit range
    print(f"\n2. Processing other samples by bit range: {len(other_samples)}")

    # Group samples by bit range
    bit_groups = defaultdict(list)
    for sample in other_samples:
        c_value = sample["c"]
        bits = get_bit_range(c_value)
        bit_groups[bits].append(sample)

    # Sort by bit range
    sorted_bits = sorted(bit_groups.keys())
    other_train = []
    other_test  = []

    for bits in sorted_bits:
        group_samples = bit_groups[bits]

        # Split each bit group by test_ratio
        if len(group_samples) == 1:
            # Only one sample, assign to training set
            group_train = group_samples
            group_test  = []
        else:
            group_train, group_test = train_test_split(
                group_samples,
                test_size=test_ratio,
                random_state=random_state
            )

        other_train.extend(group_train)
        other_test.extend(group_test)

        print(f"   Bits {bits}: {len(group_train)} train, {len(group_test)} test (total: {len(group_samples)} samples)")

    # Merge all train and test sets
    train_samples = op2_train + other_train
    test_samples  = op2_test  + other_test

    # Extract C values as targets
    train_targets = [sample["c"] for sample in train_samples]
    test_targets  = [sample["c"] for sample in test_samples]

    print(f"\nFinal Split Results:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Test samples:     {len(test_samples)}")
    print(f"  Total:            {len(train_samples) + len(test_samples)}")
    print(f"  Train/Test ratio: {len(train_samples)/(len(train_samples)+len(test_samples))*100:.2f}% / {len(test_samples)/(len(train_samples)+len(test_samples))*100:.2f}%")

    # Validation
    print(f"\n{'='*60}")
    print(f"Validation")
    print(f"{'='*60}")

    train_c_set = set(train_targets)
    test_c_set  = set(test_targets)

    print(f"Training set: {len(train_samples)} samples with {len(train_c_set)} unique C values")
    print(f"Test set:     {len(test_samples)} samples with {len(test_c_set)} unique C values")

    if len(train_targets) != len(train_c_set):
        print(f"WARNING: Training set has duplicate C values!")
        print(f"   This means the same C value appears in multiple samples.")

    if len(test_targets) != len(test_c_set):
        print(f"WARNING: Test set has duplicate C values!")
        print(f"   This means the same C value appears in multiple samples.")

    # Check for overlap between train and test C values
    overlap = train_c_set & test_c_set
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping C values between train and test!")
        print(f"   Examples: {list(overlap)[:10]}")
    else:
        print(f"No overlap between train and test C values")

    return train_targets, test_targets, train_samples, test_samples


def balanced_train_test_split(json_path, test_ratio=0.1, random_state=RANDOM_SEED):

    set_random_seed(random_state)

    train_targets, test_targets, train_samples, test_samples = custom_data_split(
        json_path, test_ratio=test_ratio, random_state=random_state
    )

    os.makedirs("./data/split", exist_ok=True)

    # Save target lists (C value lists)
    with open("./data/split/train_targets.json", "w") as f:
        json.dump(train_targets, f, indent=2)
    with open("./data/split/test_targets.json", "w") as f:
        json.dump(test_targets, f, indent=2)

    # Save full sample data
    with open("./data/split/train_data.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open("./data/split/test_data.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    # Save split info
    split_info = {
        "random_seed":           random_state,
        "test_ratio":            test_ratio,
        "train_samples_count":   len(train_samples),
        "test_samples_count":    len(test_samples),
        "train_c_values_count":  len(set(train_targets)),
        "test_c_values_count":   len(set(test_targets))
    }

    with open("./data/split/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Data split completed (Random seed: {random_state})")
    print(f"{'='*60}")
    print(f"   Train targets: ./data/split/train_targets.json ({len(train_targets)} C values)")
    print(f"   Test targets:  ./data/split/test_targets.json ({len(test_targets)} C values)")
    print(f"   Train data:    ./data/split/train_data.json ({len(train_samples)} samples)")
    print(f"   Test data:     ./data/split/test_data.json ({len(test_samples)} samples)")

    return train_targets, test_targets


def load_split_targets():
    """
    Load pre-split train and test target lists.
    Returns: train_targets, test_targets (lists of C values)
    """
    train_targets_path = "./data/split/train_targets.json"
    test_targets_path  = "./data/split/test_targets.json"

    if not os.path.exists(train_targets_path) or not os.path.exists(test_targets_path):
        print("Split targets not found, creating new split...")
        return balanced_train_test_split("./data/dpmink.json")

    with open(train_targets_path, "r") as f:
        train_targets = json.load(f)

    with open(test_targets_path, "r") as f:
        test_targets = json.load(f)

    split_info_path = "./data/split/split_info.json"
    if os.path.exists(split_info_path):
        with open(split_info_path, "r") as f:
            split_info = json.load(f)
        print(f"Loaded split targets (Random seed: {split_info['random_seed']}):")
    else:
        print(f"Loaded split targets:")

    print(f"  Training targets: {len(train_targets)} C values")
    print(f"  Test targets:     {len(test_targets)} C values")

    return train_targets, test_targets


def load_split_datasets():
    """
    Load pre-split train and test datasets.
    Returns: train_data, test_data (lists of samples)
    """
    train_data_path = "./data/split/train_data.json"
    test_data_path  = "./data/split/test_data.json"

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print("Split data not found, creating new split...")
        balanced_train_test_split("./data/dpmink.json")

    with open(train_data_path, "r") as f:
        train_data = json.load(f)

    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    split_info_path = "./data/split/split_info.json"
    if os.path.exists(split_info_path):
        with open(split_info_path, "r") as f:
            split_info = json.load(f)
        print(f"Loaded split datasets (Random seed: {split_info['random_seed']}):")
    else:
        print(f"Loaded split datasets:")

    print(f"  Training set: {len(train_data)} samples")
    print(f"  Test set:     {len(test_data)} samples")

    return train_data, test_data


if __name__ == "__main__":
    json_path = "./data/dpmink.json"

    set_random_seed(RANDOM_SEED)

    train_targets, test_targets = balanced_train_test_split(json_path)