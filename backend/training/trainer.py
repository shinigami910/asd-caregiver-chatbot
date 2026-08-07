"""
Creates the HuggingFace Trainer
and starts LoRA fine-tuning.
"""

from transformers import (

    Trainer,

    TrainingArguments,

    DataCollatorForLanguageModeling

)

from backend.training.metrics import compute_metrics

from backend.config import (

    MODEL_DIR,

    EPOCHS,

    BATCH_SIZE,

    LEARNING_RATE,

    GRADIENT_ACCUMULATION,

    WEIGHT_DECAY,

    WARMUP_STEPS,

    SAVE_STEPS,

    SAVE_TOTAL_LIMIT

)


def create_training_arguments():

    return TrainingArguments(

        output_dir=str(MODEL_DIR),

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,

        warmup_steps=WARMUP_STEPS,

        bf16=True,

        weight_decay=WEIGHT_DECAY,

        logging_steps=20,

        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        lr_scheduler_type="cosine_with_restarts",

        report_to="none",

        remove_unused_columns=False

    )


def train_model(

    model,

    tokenizer,

    tokenized_dataset

):

    training_args = create_training_arguments()

    data_collator = DataCollatorForLanguageModeling(

        tokenizer=tokenizer,

        mlm=False

    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=tokenized_dataset["train"],

        eval_dataset=tokenized_dataset["test"],

        data_collator=data_collator,

        compute_metrics=compute_metrics

    )

    print("=" * 60)

    print("Starting Fine-Tuning")

    print("=" * 60)

    trainer.train()

    print()

    print("Saving LoRA Adapter")

    model.save_pretrained(MODEL_DIR)

    tokenizer.save_pretrained(MODEL_DIR)

    print()

    print("Training Completed")

    return trainer
