from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from utils import load_env
from chains import build_generator, format_docs

class GraphState(TypedDict):
    question: str
    context_docs: List
    answer: Optional[str]
    reason: Optional[str]

def _get_retriever():
    cfg = load_env()
    embeddings = OpenAIEmbeddings(model=cfg["OPENAI_EMBEDDING_MODEL"], api_key=cfg["OPENAI_API_KEY"])
    vectordb = Chroma(
        collection_name=cfg["COLLECTION_NAME"],
        persist_directory=cfg["VECTOR_DIR"],
        embedding_function=embeddings
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})

def node_retrieve(state: GraphState) -> GraphState:
    retriever = _get_retriever()
    docs = retriever.invoke(state["question"])
    return {**state, "context_docs": docs}

def node_grade(state: GraphState) -> GraphState:
    # Simple heuristic: if no docs or docs too generic, mark reason
    docs = state.get("context_docs") or []
    if not docs:
        return {**state, "reason": "no_docs"}
    # Additional heuristic: length check on content
    lens = [len(d.page_content) for d in docs]
    if max(lens) < 20:
        return {**state, "reason": "weak_docs"}
    return {**state, "reason": None}

def node_generate(state: GraphState) -> GraphState:
    cfg = load_env()
    llm = ChatOpenAI(model=cfg["OPENAI_CHAT_MODEL"], api_key=cfg["OPENAI_API_KEY"], temperature=0.2, streaming=False)
    generator = build_generator(llm)
    context = format_docs(state.get("context_docs") or [])
    answer = generator.invoke({"question": state["question"], "context": context})
    return {**state, "answer": answer}

def node_fallback(state: GraphState) -> GraphState:
    # Friendly fallback when we couldn't retrieve context
    msg = (
        "I couldn't find a relevant FAQ entry. Could you rephrase your question? "
        "Here are a few tips: be specific about the product, order status, or policy you're asking about."
    )
    return {**state, "answer": msg}

def route_after_grade(state: GraphState):
    return "generate" if not state.get("reason") else "fallback"

def build_app_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade", node_grade)
    graph.add_node("generate", node_generate)
    graph.add_node("fallback", node_fallback)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "fallback": "fallback"})
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()