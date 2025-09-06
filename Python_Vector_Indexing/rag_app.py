from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# -------------------------------
# 0. Load environment variables
# -------------------------------
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# 1. Choose Embeddings
# -------------------------------
print("\n⚙️ Choose embedding backend:")
print("1 = HuggingFace (local, free)")
print("2 = OpenAI (cloud, requires API key)\n")

emb_choice = input("👉 Enter 1 or 2: ").strip()

if emb_choice == "2":
    if not openai_key:
        raise ValueError("❌ No OPENAI_API_KEY found in .env file")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    SAVE_PATH = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_index_openai"
    print("✅ Using OpenAI embeddings")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    SAVE_PATH = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_index_hf"
    print("✅ Using HuggingFace embeddings")

# -------------------------------
# 2. Load FAISS index
# -------------------------------
vectorstore = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

# -------------------------------
# 3. Choose LLM
# -------------------------------
print("\n🤖 Choose language model:")
print("1 = Local HuggingFace model (Gemma/TinyLlama)")
print("2 = OpenAI GPT model (requires API key)\n")

llm_choice = input("👉 Enter 1 or 2: ").strip()

if llm_choice == "2":
    if not openai_key:
        raise ValueError("❌ No OPENAI_API_KEY found in .env file")
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key, temperature=0.5)
    print("✅ Using OpenAI GPT model")
else:
    gemma_model = "google/gemma-2-2b-it"
    llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    gen_pipeline = pipeline(
        task="text-generation",
        model=gemma_model,   # swap with llama_model if needed
        device=0,
        torch_dtype="auto",
        max_new_tokens=200,
        temperature=0.5
    )
    model = HuggingFacePipeline(pipeline=gen_pipeline)
    print("✅ Using Local HuggingFace model")

# -------------------------------
# 4. Custom Prompt
# -------------------------------
template = """You are an AI assistant answering based on retrieved documents.

Context:
{context}

Question:
{question}

Answer clearly, using only the given context. 
If the answer is not in the context, say "I don't know."
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
)

# -------------------------------
# 5. Interactive loop
# -------------------------------
print("\n🤖 Vector DB Q&A App")
print("Embedding backend:", "OpenAI" if emb_choice == "2" else "HuggingFace")
print("LLM backend:", "OpenAI GPT" if llm_choice == "2" else "Local HuggingFace")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    query = input("🔎 Ask a question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting app. Goodbye!")
        break

    try:
        answer = qa.run(query)
        print(f"\n💡 Answer: {answer}\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")