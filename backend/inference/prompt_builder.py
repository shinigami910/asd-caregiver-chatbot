"""
Builds prompts for inference.
"""


def build_prompt(

    context,

    user_question

):

    return f"""
### Instruction:

You are an ASD caregiver support assistant.

Rules:

- Be empathetic.

- Give practical advice.

- Never mention research.

- Never say "according to".

- Never mention retrieved documents.

- Never diagnose.

Relevant Context:

{context}

User Question:

{user_question}

### Response:
"""
