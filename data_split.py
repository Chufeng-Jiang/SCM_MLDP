"""
data_split.py

Reproducible train/test splitting utility for SCM decomposition datasets.

This script implements a custom split strategy for JSON-formatted structured
causal model (SCM) samples. Each sample is expected to be a complete JSON object
of the form:

    {
        "c": <target integer>,
        "equations": [ ... decomposition steps ... ]
    }

The splitting protocol is designed for research experiments where leakage
between train and test sets must be avoided at the target-value level. In
particular, the same C value should not appear in both training and test sets.
The script also preserves useful distributional structure by grouping non-OP=2
samples according to the bit length of C before applying the train/test split.
"""

import json
import os
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split


# Global seed used to make all random operations reproducible.
RANDOM_SEED = 42


def set_random_seed(seed=RANDOM_SEED):
    """
    Set random seeds for reproducible experiments.

    This function seeds NumPy and, when available, PyTorch. PyTorch is imported
    lazily so that this utility can still be used in environments where PyTorch
    is not installed.

    Args:
        seed (int): Random seed used by NumPy and PyTorch.
    """
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # PyTorch is optional for this data-splitting script.
        pass


def get_bit_range(c_value):
    """
    Return the bit length of a target integer C.

    The bit length is used as a coarse complexity bucket. Grouping samples by
    bit length helps keep the train/test split more balanced across small and
    large target values.

    Args:
        c_value (int): Target integer C.

    Returns:
        int: Number of bits required to represent C in binary.
    """
    if c_value == 0:
        return 0

    return c_value.bit_length()


def custom_data_split(json_path, test_ratio=0.1, random_state=RANDOM_SEED):
    """
    Create a custom train/test split based on target value C and bit length.

    Splitting strategy:
        1. Load the full JSON dataset.
        2. Check whether each target C value is unique.
        3. Separate samples that contain OP=2 from the rest.
        4. Randomly assign up to 10,000 OP=2 samples to the training set.
        5. Group all remaining samples by the bit length of C.
        6. Split each bit-length group into train/test subsets.
        7. Validate that no C value appears in both train and test sets.

    This design is useful when OP=2 is relatively important or sparse and should
    be sufficiently represented in the training set. The bit-length grouping also
    reduces the chance that the test set is dominated by a narrow numerical range.

    Args:
        json_path (str): Path to the input JSON dataset.
        test_ratio (float): Proportion of each non-OP=2 bit group assigned to test.
        random_state (int): Seed used for reproducible splitting.

    Returns:
        tuple:
            train_targets (list): C values assigned to the training set.
            test_targets (list): C values assigned to the test set.
            train_samples (list): Full JSON samples assigned to training.
            test_samples (list): Full JSON samples assigned to testing.
    """
    print("=" * 60)
    print("Custom Data Split Strategy")
    print("=" * 60)

    set_random_seed(random_state)

    # Load the full dataset from disk.
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    print(f"Total samples in dataset: {len(raw_data)}")

    # Verify whether C values are unique across samples. If duplicate C values
    # exist, the dataset may contain multiple decomposition traces for the same
    # target value, which increases the risk of train/test leakage.
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
        print("   This suggests that the dataset may contain multiple decompositions for the same C value.")
    else:
        print(f"All C values are unique ({len(c_values_seen)} unique C values)")

    # Partition samples according to whether their equation sequence contains
    # operation type OP=2. This allows OP=2 cases to be controlled explicitly.
    op2_samples = []
    other_samples = []

    for sample in raw_data:
        equations = sample["equations"]

        # A sample is considered OP=2-related if any equation uses op == 2.
        has_op2 = any(eq.get("op", 0) == 2 for eq in equations)

        if has_op2:
            op2_samples.append(sample)
        else:
            other_samples.append(sample)

    print("\nSample categories:")
    print(f"  Samples with OP=2: {len(op2_samples)}")
    print(f"  Other samples: {len(other_samples)}")

    # Step 1: Handle OP=2 samples separately.
    # If there are more than 10,000 OP=2 samples, randomly select 10,000 for
    # training and place the rest in the test set. If there are 10,000 or fewer,
    # keep all OP=2 samples in training to maximize coverage of this operation.
    print(f"\n1. Processing samples with OP=2: {len(op2_samples)}")
    rng = np.random.RandomState(random_state)

    if len(op2_samples) > 10000:
        indices = rng.permutation(len(op2_samples))
        op2_train = [op2_samples[i] for i in indices[:10000]]
        op2_test = [op2_samples[i] for i in indices[10000:]]
    else:
        op2_train = op2_samples
        op2_test = []

    print(f"   OP=2 samples: {len(op2_train)} train, {len(op2_test)} test")

    # Step 2: Split all non-OP=2 samples by bit-length buckets.
    print(f"\n2. Processing other samples by bit range: {len(other_samples)}")

    bit_groups = defaultdict(list)
    for sample in other_samples:
        c_value = sample["c"]
        bits = get_bit_range(c_value)
        bit_groups[bits].append(sample)

    sorted_bits = sorted(bit_groups.keys())
    other_train = []
    other_test = []

    for bits in sorted_bits:
        group_samples = bit_groups[bits]

        # Split each bit-length group independently. If a bucket contains only
        # one sample, it is assigned to training to avoid an invalid split.
        if len(group_samples) == 1:
            group_train = group_samples
            group_test = []
        else:
            group_train, group_test = train_test_split(
                group_samples,
                test_size=test_ratio,
                random_state=random_state,
            )

        other_train.extend(group_train)
        other_test.extend(group_test)

        print(
            f"   Bits {bits}: {len(group_train)} train, "
            f"{len(group_test)} test (total: {len(group_samples)} samples)"
        )

    # Combine OP=2-controlled samples with bit-length-balanced samples.
    train_samples = op2_train + other_train
    test_samples = op2_test + other_test

    # Store target C values separately. These target lists can be used later by
    # dataset loaders to filter samples without re-running the split procedure.
    train_targets = [sample["c"] for sample in train_samples]
    test_targets = [sample["c"] for sample in test_samples]

    print("\nFinal Split Results:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Test samples:     {len(test_samples)}")
    print(f"  Total:            {len(train_samples) + len(test_samples)}")
    print(
        f"  Train/Test ratio: "
        f"{len(train_samples) / (len(train_samples) + len(test_samples)) * 100:.2f}% / "
        f"{len(test_samples) / (len(train_samples) + len(test_samples)) * 100:.2f}%"
    )

    # Validation checks for reproducibility and leakage prevention.
    print(f"\n{'=' * 60}")
    print("Validation")
    print(f"{'=' * 60}")

    train_c_set = set(train_targets)
    test_c_set = set(test_targets)

    print(f"Training set: {len(train_samples)} samples with {len(train_c_set)} unique C values")
    print(f"Test set:     {len(test_samples)} samples with {len(test_c_set)} unique C values")

    if len(train_targets) != len(train_c_set):
        print("WARNING: Training set has duplicate C values!")
        print("   This means the same C value appears in multiple training samples.")

    if len(test_targets) != len(test_c_set):
        print("WARNING: Test set has duplicate C values!")
        print("   This means the same C value appears in multiple test samples.")

    # The most important validation step: ensure no C value appears in both
    # train and test sets.
    overlap = train_c_set & test_c_set
    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping C values between train and test!")
        print(f"   Examples: {list(overlap)[:10]}")
    else:
        print("No overlap between train and test C values")

    return train_targets, test_targets, train_samples, test_samples


def balanced_train_test_split(json_path, test_ratio=0.1, random_state=RANDOM_SEED):
    """
    Run the custom split procedure and persist the resulting files.

    This function saves both compact target lists and full sample-level datasets.
    The compact target lists are useful for filtering the original dataset during
    model training, while the full split files provide a fixed experimental
    record for reproducibility.

    Args:
        json_path (str): Path to the input JSON dataset.
        test_ratio (float): Proportion of non-OP=2 samples assigned to the test set.
        random_state (int): Random seed used for reproducibility.

    Returns:
        tuple: train_targets and test_targets.
    """
    set_random_seed(random_state)

    train_targets, test_targets, train_samples, test_samples = custom_data_split(
        json_path,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # Create the output directory if it does not already exist.
    os.makedirs("./data/split", exist_ok=True)

    # Save target lists containing only C values.
    with open("./data/split/train_targets.json", "w") as f:
        json.dump(train_targets, f, indent=2)

    with open("./data/split/test_targets.json", "w") as f:
        json.dump(test_targets, f, indent=2)

    # Save full sample-level datasets for reproducible experiments.
    with open("./data/split/train_data.json", "w") as f:
        json.dump(train_samples, f, indent=2)

    with open("./data/split/test_data.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    # Save metadata describing the split configuration.
    split_info = {
        "random_seed": random_state,
        "test_ratio": test_ratio,
        "train_samples_count": len(train_samples),
        "test_samples_count": len(test_samples),
        "train_c_values_count": len(set(train_targets)),
        "test_c_values_count": len(set(test_targets)),
    }

    with open("./data/split/split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Data split completed (Random seed: {random_state})")
    print(f"{'=' * 60}")
    print(f"   Train targets: ./data/split/train_targets.json ({len(train_targets)} C values)")
    print(f"   Test targets:  ./data/split/test_targets.json ({len(test_targets)} C values)")
    print(f"   Train data:    ./data/split/train_data.json ({len(train_samples)} samples)")
    print(f"   Test data:     ./data/split/test_data.json ({len(test_samples)} samples)")

    return train_targets, test_targets


def load_split_targets():
    """
    Load precomputed train/test target lists.

    If the split files do not exist, the function automatically creates a new
    split using the default dataset path. This makes downstream training scripts
    simpler because they can rely on this function to return valid target lists.

    Returns:
        tuple:
            train_targets (list): C values used for training.
            test_targets (list): C values used for testing.
    """
    train_targets_path = "./data/split/train_targets.json"
    test_targets_path = "./data/split/test_targets.json"

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
        print("Loaded split targets:")

    print(f"  Training targets: {len(train_targets)} C values")
    print(f"  Test targets:     {len(test_targets)} C values")

    return train_targets, test_targets


def load_split_datasets():
    """
    Load precomputed full train/test datasets.

    Unlike load_split_targets(), this function returns the complete JSON samples,
    including both the target C value and its equation sequence. This is useful
    for analysis, debugging, and verifying sample-level properties.

    Returns:
        tuple:
            train_data (list): Full JSON samples used for training.
            test_data (list): Full JSON samples used for testing.
    """
    train_data_path = "./data/split/train_data.json"
    test_data_path = "./data/split/test_data.json"

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
        print("Loaded split datasets:")

    print(f"  Training set: {len(train_data)} samples")
    print(f"  Test set:     {len(test_data)} samples")

    return train_data, test_data


if __name__ == "__main__":
    # Default dataset path used by the experiment pipeline.
    json_path = "./data/dpmink.json"

    # Ensure deterministic behavior before creating the split.
    set_random_seed(RANDOM_SEED)

    # Generate and save the fixed train/test split.
    train_targets, test_targets = balanced_train_test_split(json_path)
