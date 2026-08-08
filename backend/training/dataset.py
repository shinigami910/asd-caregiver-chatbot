import json
from datasets import Dataset
from backend.config import (
    TEST_SPLIT,
    TRAIN_DATASET,
    TEST_DATASET
)


def load_instruction_dataset(file_path):
    """
    Reads instruction-output JSONL dataset.

    Expected format:

    {
        "instruction": "...",
        "output": "..."
    }
    """

    samples = []

    with open(file_path, "r", encoding="utf-8") as f:

        for lineno, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

            except json.JSONDecodeError as e:

                print(f"Skipping invalid JSON at line {lineno}: {e}")

                continue

            instruction = obj.get("instruction", "").strip()

            output = obj.get("output", "").strip()

            if not instruction or not output:

                print(f"Skipping incomplete sample at line {lineno}")

                continue

            samples.append({

                "instruction": instruction,

                "output": output

            })

    return Dataset.from_list(samples)


def prepare_dataset(input_file):
    """
    Creates train/test split
    and saves both datasets.
    """

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    dataset = load_instruction_dataset(input_file)

    print(f"Loaded {len(dataset):,} samples")

    dataset = dataset.shuffle(seed=42)

    split = dataset.train_test_split(
        test_size=TEST_SPLIT
    )

    print(f"Training Samples : {len(split['train']):,}")

    print(f"Testing Samples  : {len(split['test']):,}")

    split["train"].to_json(
        TRAIN_DATASET,
        orient="records",
        lines=True
    )

    split["test"].to_json(
        TEST_DATASET,
        orient="records",
        lines=True
    )

    print()

    print("Dataset successfully prepared.")

    return split
