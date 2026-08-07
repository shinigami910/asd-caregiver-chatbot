"""
Generate answers using the LLM.
"""

import torch

from backend.config import (
    MAX_NEW_TOKENS
)

from backend.inference.prompt_builder import (
    build_prompt
)


class ResponseGenerator:

    def __init__(

        self,

        model,

        tokenizer,

        retriever

    ):

        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever

    def generate(

        self,

        question

    ):

        context = self.retriever.retrieve(
            question
        )

        prompt = build_prompt(

            context,

            question

        )

        inputs = self.tokenizer(

            prompt,

            return_tensors="pt",

            truncation=True,

            max_length=2048

        ).to(

            self.model.device

        )

        with torch.inference_mode():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=False,

                pad_token_id=self.tokenizer.eos_token_id

            )

        decoded = self.tokenizer.decode(

            outputs[0],

            skip_special_tokens=True

        )

        if "### Response:" in decoded:

            decoded = decoded.split(
                "### Response:"
            )[-1].strip()

        return decoded
