from typing import List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_rag_prompt():
    system = (
        "You are a helpful customer-support assistant. "
        "Use the retrieved FAQ context to answer the user's question. "
        "If the answer isn't in the context, say you don't know and suggest the closest relevant info."
    )
    template = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely and cite sources as [#].")
    ])
    return template

def format_docs(docs: List[Any]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        parts.append(f"[{i}] {d.page_content}")
    return "\n\n".join(parts)

def build_generator(llm):
    prompt = build_rag_prompt()
    chain = prompt | llm | StrOutputParser()
    return chain