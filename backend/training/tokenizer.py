"""
1. Load tokenizer
2. Setting padding token
3. Configuring padding strategy
"""
from transformers import AutoTokenizer
from backend.config import MODEL_NAME

def load_tokenizer():

    print("=" * 60)
    print("Loading Tokenizer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Mistral has no default pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Right padding is required for causal LM training
    tokenizer.padding_side = "right"

    print("Tokenizer loaded successfully.")

    return tokenizer
