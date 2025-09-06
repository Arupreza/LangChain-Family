# ================================
# PDF → OpenAI Embeddings → FAISS Index
# ================================
# Loads PDFs, chunks text, embeds with OpenAI, saves FAISS index,
# and runs similarity search. API key is read from `.env`.
# ================================

from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# -------------------------------
# 0) Load environment variables
# -------------------------------
load_dotenv()  # reads .env file in the same directory or project root
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

print("✅ OpenAI API key loaded from .env")

# -------------------------------
# 1) Config
# -------------------------------
DATA_DIR = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/Data"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
SAVE_PATH = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_openai_index"
EMBED_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"

# -------------------------------
# 2) Load PDF documents
# -------------------------------
print("🔎 Step 1: Loading PDF documents...")
loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} PDF-documents.")

if not documents:
    raise ValueError(f"No PDF documents found in {DATA_DIR}")

# -------------------------------
# 3) Split into chunks
# -------------------------------
print("\n🔎 Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks.")
print("\n🔎 Example chunk:")
print(chunks[0].page_content[:200], "...")

# -------------------------------
# 4) Create OpenAI embeddings
# -------------------------------
print("\n🔎 Step 3: Generating embeddings with OpenAI...")
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)

# -------------------------------
# 5) Build FAISS index
# -------------------------------
print("🔎 Step 4: Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
print("✅ FAISS index built.")

# -------------------------------
# 6) Save FAISS index
# -------------------------------
print("\n🔎 Step 5: Saving index...")
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
vectorstore.save_local(SAVE_PATH)
print(f"✅ Saved FAISS index to {SAVE_PATH}")

# -------------------------------
# 7) Reload FAISS index
# -------------------------------
print("\n🔎 Step 6: Reloading index...")
new_store = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
print("✅ Reloaded FAISS index.")

# -------------------------------
# 8) Run similarity search
# -------------------------------
print("\n🔎 Step 7: Querying...")
query = "intrusion detection in CAN bus"
results = new_store.similarity_search(query, k=3)

print("\n🔎 Query Results:")
for r in results:
    print(f"- {r.page_content[:120]} ... (source={r.metadata.get('source')})")