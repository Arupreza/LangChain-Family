# app.py
# ----------------------------------------------------
# Vector DB Q&A (FAISS + Compression + Conversational Memory)
# End-to-end Streamlit app (single file)
# ----------------------------------------------------
from pathlib import Path
import os
import traceback
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# LangChain + retriever stack
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# HF pipeline for local LLMs
from transformers import pipeline

# -------------------------------
# 0) Page config
# -------------------------------
st.set_page_config(page_title="Vector DB Indexed Q&A App", page_icon="🧠", layout="wide")

# -------------------------------
# 1) Env + sidebar controls
# -------------------------------
load_dotenv()
st.sidebar.header("⚙️ Configuration")

openai_key_env = os.getenv("OPENAI_API_KEY", "")
openai_key_ui = st.sidebar.text_input("OpenAI API Key", value=openai_key_env, type="password", help="Only needed for OpenAI embeddings/LLM")
openai_key = openai_key_ui or openai_key_env

st.sidebar.markdown("---")
emb_choice = st.sidebar.selectbox(
    "Embedding backend / index type",
    ["HuggingFace (Flat)", "OpenAI (Flat)", "OpenAI (HNSW)"],
    index=0
)

default_paths = {
    "HuggingFace (Flat)": "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_index_hf",
    "OpenAI (Flat)": "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_index_openai",
    "OpenAI (HNSW)": "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_openai_index_hnsw",
}
index_path = st.sidebar.text_input("FAISS index folder", value=default_paths[emb_choice])

k = st.sidebar.slider("Top-k (base retriever)", min_value=1, max_value=20, value=5, step=1)

st.sidebar.markdown("---")
llm_choice = st.sidebar.selectbox("Language model", ["Local HuggingFace", "OpenAI GPT"], index=0)
temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.5, 0.1)

hf_model = st.sidebar.selectbox(
    "Local HF model (if 'Local HuggingFace')",
    ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "google/gemma-2-2b-it"],
    index=0
)
use_cuda = st.sidebar.checkbox("Use GPU (CUDA:0 if available)", value=True)

st.sidebar.markdown("---")
use_history = st.sidebar.toggle("Use chat history for answers", value=True)
show_sources = st.sidebar.toggle("Show retrieved chunks (sources)", value=False)

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.pop("messages", None)
    st.session_state.pop("serialized_history", None)
    st.experimental_rerun()

# -------------------------------
# 2) Prompt
# -------------------------------
PROMPT_TMPL = """You are an AI assistant answering based on retrieved documents and the ongoing conversation.

Chat history:
{chat_history}

Context:
{context}

Question:
{question}

Answer clearly, using only the given context. If the answer is not in the context, say "I don't know."
"""
QA_PROMPT = PromptTemplate(
    template=PROMPT_TMPL,
    input_variables=["chat_history", "context", "question"],
)

# -------------------------------
# 3) Helpers (cached + robust)
# -------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings(emb_choice: str, openai_key: str):
    """Return an embeddings object. Args are hashable, return value is cached as a resource."""
    if emb_choice.startswith("OpenAI"):
        if not openai_key:
            raise ValueError("No OpenAI API key provided for OpenAI embeddings.")
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_faiss_index(path: str, _embeddings):
    """
    Load FAISS index. NOTE: `_embeddings` has a leading underscore so Streamlit
    doesn't try to hash it (non-hashable object).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS index folder not found: {p}")
    return FAISS.load_local(str(p), _embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def get_llm(llm_choice: str, temperature: float, openai_key: str, hf_model: str, use_cuda: bool):
    """Return an LLM (either OpenAI chat model or local HF pipeline wrapped via HuggingFacePipeline)."""
    if llm_choice == "OpenAI GPT":
        if not openai_key:
            raise ValueError("No OpenAI API key provided for OpenAI LLM.")
        # Change to e.g. 'gpt-4o-mini' if you prefer
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key, temperature=temperature)

    device = 0 if use_cuda else -1
    gen_pipe = pipeline(
        task="text-generation",
        model=hf_model,
        device=device,
        torch_dtype="auto",
        max_new_tokens=200,
        temperature=temperature,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

def build_memory_from_session():
    """Create a ConversationBufferMemory and seed it with prior turns if present."""
    mem = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    # Rehydrate from serialized history (list of dicts: {"role","content"})
    history: List[Dict[str, str]] = st.session_state.get("serialized_history", [])
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            mem.chat_memory.add_user_message(content)
        elif role == "assistant":
            mem.chat_memory.add_ai_message(content)
    return mem

def serialize_turn(role: str, content: str):
    """Append a single turn to session history for future rebuilds."""
    if "serialized_history" not in st.session_state:
        st.session_state.serialized_history = []
    st.session_state.serialized_history.append({"role": role, "content": content})

def ensure_chat_ui_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask a question about your indexed corpus."}]

def build_conv_chain(
    emb_choice: str,
    index_path: str,
    k: int,
    llm_choice: str,
    temperature: float,
    openai_key: str,
    hf_model: str,
    use_cuda: bool,
    use_history: bool,
    show_sources: bool,
):
    """
    Build the ConversationalRetrievalChain on top of
    FAISS similarity → LLM-based contextual compression retriever.
    """
    embeddings = get_embeddings(emb_choice, openai_key)
    vectorstore = load_faiss_index(index_path, embeddings)

    base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    llm = get_llm(llm_choice, temperature, openai_key, hf_model, use_cuda)

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )

    memory = build_memory_from_session() if use_history else None

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=show_sources,
        verbose=False,
    )
    return chain

# -------------------------------
# 4) Header + chain build
# -------------------------------
st.title("🧠 Vector DB Q&A (FAISS + Compression + Conversational Memory)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Embedding backend", emb_choice)
with col2:
    st.metric("Index path", Path(index_path).name or index_path)
with col3:
    st.metric("LLM", llm_choice)

st.divider()

with st.spinner("Building retrieval chain..."):
    try:
        qa = build_conv_chain(
            emb_choice=emb_choice,
            index_path=index_path,
            k=k,
            llm_choice=llm_choice,
            temperature=temperature,
            openai_key=openai_key,
            hf_model=hf_model,
            use_cuda=use_cuda,
            use_history=use_history,
            show_sources=show_sources,
        )
    except Exception as e:
        st.error(f"Failed to initialize chain: {e}")
        st.code(traceback.format_exc())
        st.stop()

# -------------------------------
# 5) Chat UI (history-aware)
# -------------------------------
ensure_chat_ui_state()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_query = st.chat_input("Type your question…")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    serialize_turn("user", user_query)
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving with compression & using chat history…" if use_history else "Retrieving with compression…"):
            try:
                # ConversationalRetrievalChain expects {"question": "..."}
                result: Dict[str, Any] = qa.invoke({"question": user_query})  # type: ignore
                answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        serialize_turn("assistant", answer)

        # Optional: show source documents/chunks
        if show_sources and isinstance(result, dict) and "source_documents" in result:
            src_docs = result.get("source_documents") or []
            with st.expander(f"🔎 Retrieved Chunks ({len(src_docs)})"):
                for i, d in enumerate(src_docs, 1):
                    # Try to print page/range metadata if available
                    meta = d.metadata if hasattr(d, "metadata") else {}
                    st.markdown(f"**Chunk {i}**")
                    if meta:
                        st.code(meta)
                    st.write(d.page_content)

st.caption(
    "Pipeline: FAISS similarity(k) → LLM-based contextual compression → ConversationalRetrievalChain with optional chat history. "
    "If info isn’t in retrieved context, the model says “I don't know.”"
)
