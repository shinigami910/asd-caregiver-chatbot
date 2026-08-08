"""
1. Loading base model
2. Applying LoRA
3. Printing trainable parameters
"""
import torch
from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model
)
from backend.config import (
    MODEL_NAME,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT
)

def load_lora_model():

    print("=" * 60)
    print("Loading Base Model")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("Applying LoRA...")

    lora_config = LoraConfig(

        r=LORA_RANK,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],

        task_type="CAUSAL_LM"

    )

    model = get_peft_model(
        model,
        lora_config
    )

    print_trainable_parameters(model)

    return model


def print_trainable_parameters(model):

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    percentage = (
        100 * trainable_params / total_params
    )

    print()

    print(
        f"Trainable Parameters : "
        f"{trainable_params:,}"
    )

    print(
        f"Total Parameters : "
        f"{total_params:,}"
    )

    print(
        f"Trainable Percentage : "
        f"{percentage:.2f}%"
    )
