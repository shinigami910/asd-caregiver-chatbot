"""
1. Formatting prompts
2. Tokenizing dataset
3. Preparing datasets for training
"""
from backend.config import MAX_LENGTH
def format_instruction(instruction, output, tokenizer):
    """
    Converts one instruction-output pair
    into the prompt used for fine-tuning.
    """

    return (
        f"### Instruction:\n"
        f"{instruction}\n\n"
        f"### Response:\n"
        f"{output}"
        f"{tokenizer.eos_token}"
    )


def tokenize_batch(batch, tokenizer):

    formatted = [

        format_instruction(
            instruction,
            output,
            tokenizer
        )

        for instruction, output in zip(
            batch["instruction"],
            batch["output"]
        )

    ]

    outputs = tokenizer(

        formatted,

        truncation=True,

        max_length=MAX_LENGTH,

        padding="max_length"

    )

    # Labels for causal LM
    outputs["labels"] = outputs["input_ids"].copy()

    return outputs


def prepare_tokenized_dataset(dataset, tokenizer):
    """
    Converts the raw dataset
    into a tokenized dataset.
    """

    tokenized_dataset = dataset.map(

        lambda batch: tokenize_batch(
            batch,
            tokenizer
        ),

        batched=True,

        remove_columns=dataset["train"].column_names

    )

    return tokenized_dataset
