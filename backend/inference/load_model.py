"""
Loads the base model and LoRA adapter.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

from backend.config import (
    MODEL_NAME,
    MODEL_DIR
)


class ASDModel:

    def __init__(self):

        self.tokenizer = None
        self.model = None
        self.device = None

    def load(self):

        print("=" * 60)
        print("Loading Tokenizer")
        print("=" * 60)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        print("=" * 60)
        print("Loading Base Model")
        print("=" * 60)

        base_model = AutoModelForCausalLM.from_pretrained(

            MODEL_NAME,

            device_map="auto",

            torch_dtype=torch.bfloat16

        )

        print("=" * 60)
        print("Loading LoRA Adapter")
        print("=" * 60)

        self.model = PeftModel.from_pretrained(

            base_model,

            MODEL_DIR

        )

        self.model.eval()

        self.device = next(
            self.model.parameters()
        ).device

        print(f"Model Loaded on {self.device}")

        return self
