# Tiny offline eval to showcase LangSmith-compatible workflows.
# Computes (1) retrieval recall@k for the exact question reconstruction;
# and (2) a simplistic answer match based on substring checks.

from datasets import load_dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from chains import build_generator, format_docs
from utils import load_env
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def main():
    cfg = load_env()
    ds = load_dataset(cfg["HF_DATASET_PATH"])["train"]

    embeddings = OpenAIEmbeddings(model=cfg["OPENAI_EMBEDDING_MODEL"], api_key=cfg["OPENAI_API_KEY"])
    vectordb = Chroma(collection_name=cfg["COLLECTION_NAME"], persist_directory=cfg["VECTOR_DIR"], embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model=cfg["OPENAI_CHAT_MODEL"], api_key=cfg["OPENAI_API_KEY"], temperature=0.0)
    gen = build_generator(llm)

    y_true, y_pred = [], []
    hits, total = 0, 0

    for ex in tqdm(ds, desc="Evaluating"):
        q = ex["question"].strip()
        a = ex["answer"].strip()
        docs = retriever.invoke(q)
        total += 1
        # Simple recall@k: if any retrieved doc contains the gold answer string
        if any(a[:40].lower() in d.page_content.lower() for d in docs):
            hits += 1

        context = format_docs(docs)
        out = gen.invoke({"question": q, "context": context})
        # Crude correctness: does the model output include a key answer span?
        y_true.append(1)
        y_pred.append(1 if a[:40].lower() in out.lower() else 0)

    print(f"Retrieval Recall@{len(docs)}: {hits/total:.3f}")
    print("Answer match (naive):", accuracy_score(y_true, y_pred))

if __name__ == "__main__":
    main()