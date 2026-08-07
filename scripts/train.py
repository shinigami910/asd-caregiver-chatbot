"""
train.py

Complete training pipeline.
"""

from backend.config import RAW_DATASET

from backend.training import (

    prepare_dataset,

    load_tokenizer,

    load_lora_model,

    prepare_tokenized_dataset,

    train_model

)


def main():

    print("=" * 70)
    print("ASD MODEL TRAINING")
    print("=" * 70)

    dataset = prepare_dataset(RAW_DATASET)

    tokenizer = load_tokenizer()

    model = load_lora_model()

    tokenized_dataset = prepare_tokenized_dataset(

        dataset,

        tokenizer

    )

    train_model(

        model,

        tokenizer,

        tokenized_dataset

    )

    print()

    print("Training Complete")


if __name__ == "__main__":

    main()
