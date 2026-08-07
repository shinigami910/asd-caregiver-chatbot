"""
merge_lora.py
"""

from peft import PeftModel

from transformers import (

    AutoModelForCausalLM,

    AutoTokenizer

)

from backend.config import (

    MODEL_NAME,

    MODEL_DIR

)


def main():

    tokenizer = AutoTokenizer.from_pretrained(

        MODEL_DIR

    )

    base_model = AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        device_map="auto"

    )

    model = PeftModel.from_pretrained(

        base_model,

        MODEL_DIR

    )

    merged = model.merge_and_unload()

    merged.save_pretrained(

        MODEL_DIR / "merged"

    )

    tokenizer.save_pretrained(

        MODEL_DIR / "merged"

    )

    print("Merged Model Saved")


if __name__ == "__main__":

    main()
