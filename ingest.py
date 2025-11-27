import os
from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from utils import load_env

def main():
    cfg = load_env()
    os.makedirs(cfg["VECTOR_DIR"], exist_ok=True)

    print("Loading dataset:", cfg["HF_DATASET_PATH"])
    ds = load_dataset(cfg["HF_DATASET_PATH"])["train"]

    # Convert Q/A rows → docs (we'll store both Q & A in page_content for semantic retrieval)
    docs = []
    for i, row in enumerate(ds):
        q = row["question"].strip()
        a = row["answer"].strip()
        content = f"Question: {q}\nAnswer: {a}"
        docs.append(Document(page_content=content, metadata={"id": i, "source": "hf_faq"}))

    print(f"Loaded {len(docs)} items")

    embeddings = OpenAIEmbeddings(
        model=cfg["OPENAI_EMBEDDING_MODEL"],
        api_key=cfg["OPENAI_API_KEY"]
    )

    print("Building / updating Chroma collection:", cfg["COLLECTION_NAME"])
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=cfg["COLLECTION_NAME"],
        persist_directory=cfg["VECTOR_DIR"],
    )
    vectordb.persist()
    print("Done. Vector DB persisted at:", cfg["VECTOR_DIR"])

if __name__ == "__main__":
    main()