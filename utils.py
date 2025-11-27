import os
from dotenv import load_dotenv

# Shared helpers

def load_env():
    # Load .env once and return key config values
    load_dotenv(override=True)
    cfg = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "HF_DATASET_PATH": os.getenv("HF_DATASET_PATH", "MakTek/Customer_support_faqs_dataset"),
        "VECTOR_DIR": os.getenv("VECTOR_DIR", "./vectorstore"),
        "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "faq_collection"),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", "false").lower() in ("1","true","yes","y"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY", ""),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "customer-support-faqs-demo"),
    }
    return cfg