# ================================
# PDF → Embeddings → FAISS Index
# ================================
# This script loads PDFs, chunks them into text passages,
# embeds them with a HuggingFace model, stores them in a FAISS index,
# and then runs a similarity search query.
# ================================

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# ------------ CONFIG ------------
DATA_DIR = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/Data"  # folder containing PDFs
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"                   # embedding model (384-dim)
CHUNK_SIZE = 800        # max characters per text chunk
CHUNK_OVERLAP = 120     # overlap between chunks (to preserve context)
SAVE_PATH = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_index"
# --------------------------------

# 1) Load PDF documents
print("🔎 Step 1: Loading PDF documents...")
loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.pdf",      # recursively look for all PDFs
    loader_cls=PyPDFLoader,  # load each PDF as a set of LangChain Documents
    show_progress=True,
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} PDF-documents.")

if not documents:
    raise ValueError(f"No PDF documents found in {DATA_DIR}")

# 2) Split into chunks
print("\n🔎 Step 2: Splitting into chunks...")
# Large PDF pages can be too big for embedding models,
# so we split them into smaller overlapping chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,       # each chunk up to 800 chars
    chunk_overlap=CHUNK_OVERLAP, # chunks overlap by 120 chars for continuity
)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks.")

# Show an example chunk for verification
print("\n🔎 Example chunk:")
print(chunks[0].page_content[:200], "...")

# 3) Create embedding model
print("\n🔎 Step 3: Embeddings...")
# HuggingFace wrapper for sentence-transformers model
# "all-MiniLM-L6-v2" is small & fast, good for semantic similarity
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# 4) Build FAISS index
print("🔎 Step 4: Building FAISS index...")
# LangChain’s FAISS wrapper:
# takes Documents + embeddings → builds FAISS vector store
vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
print("✅ FAISS index built.")

# 5) Save FAISS index to disk
print("\n🔎 Step 5: Saving index...")
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)  # ensure save dir exists
vectorstore.save_local(SAVE_PATH)
print(f"✅ Saved FAISS index to {SAVE_PATH}")

# 6) Reload FAISS index from disk
print("\n🔎 Step 6: Reloading index...")
# When reloading, you must pass the same embedding function object
new_store = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
print("✅ Reloaded FAISS index.")

# 7) Run a similarity search query
print("\n🔎 Step 7: Querying...")
query = "intrusion detection in CAN bus"
results = new_store.similarity_search(query, k=3)

# Show top 3 retrieved chunks
print("\n🔎 Query Results:")
for r in results:
    print(f"- {r.page_content[:120]} ... (source={r.metadata.get('source')})")