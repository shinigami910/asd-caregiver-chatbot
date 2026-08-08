from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RAG_DIR = PROJECT_ROOT / "rag"

# Base Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Dataset
RAW_DATASET = DATA_DIR / "raw" / "asdsupport_raw.jsonl"
TRAIN_DATASET = DATA_DIR / "processed" / "asdsupport_train.jsonl"
TEST_DATASET = DATA_DIR / "processed" / "asdsupport_test.jsonl"
TEST_SPLIT = 0.10

# LoRA
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# Tokenization
MAX_LENGTH = 1024

# Training
EPOCHS = 5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION = 16
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2

# RAG
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Generation
MAX_NEW_TOKENS = 300
