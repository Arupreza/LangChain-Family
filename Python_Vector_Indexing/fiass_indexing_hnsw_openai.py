import faiss
from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.docstore.in_memory import InMemoryDocstore

# -------------------------------
# 0) Load environment variables
# -------------------------------
# Reads your `.env` file where the OpenAI API key is stored
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

print("✅ API key loaded")

# -------------------------------
# 1) Config
# -------------------------------
# Define paths and parameters for indexing
DATA_DIR = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/Data"  # where PDFs are located
SAVE_PATH = "/home/lisa/Arupreza/LangChain/Python_Vector_Indexing/faiss_openai_index_hnsw"  # where index will be saved
CHUNK_SIZE = 800        # max characters per chunk
CHUNK_OVERLAP = 120     # overlap between chunks to keep context
EMBED_MODEL = "text-embedding-3-small"  # OpenAI embedding model (dim=1536)

# -------------------------------
# 2) Load PDFs
# -------------------------------
print("🔎 Step 1: Loading PDFs...")
# DirectoryLoader recursively loads all PDFs in the data directory
loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

if not documents:
    raise ValueError("❌ No PDFs found")

# -------------------------------
# 3) Split into chunks
# -------------------------------
print("🔎 Step 2: Splitting into chunks...")
# Large PDF pages may be too big → split into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks")

# -------------------------------
# 4) Embeddings
# -------------------------------
print("🔎 Step 3: Creating embeddings...")
# Use OpenAI embeddings model to convert text into vectors
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)

# Create embedding vectors for each text chunk
texts = [c.page_content for c in chunks]  # raw text
metadatas = [c.metadata for c in chunks]  # metadata (e.g., filename, page)
vectors = embeddings.embed_documents(texts)  # actual vectors
dim = len(vectors[0])  # embedding dimension (1536 for OpenAI models)

# -------------------------------
# 5) Create HNSW Index
# -------------------------------
print("🔎 Step 4: Building HNSW index...")
# HNSW parameters:
M = 64                # number of neighbors per node
ef_construction = 200 # build-time accuracy
ef_search = 50        # query-time accuracy

# Build FAISS HNSW index
index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = ef_search

# -------------------------------
# 6) Wrap FAISS into LangChain Vectorstore
# -------------------------------
# LangChain’s FAISS wrapper requires:
# - the FAISS index
# - a docstore (mapping IDs → documents)
# - an index_to_docstore_id mapping
docstore = InMemoryDocstore()
index_to_docstore_id = {}

vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Add embeddings + metadata into FAISS index
vectorstore.add_texts(texts, metadatas=metadatas)

print("✅ HNSW FAISS index built")

# -------------------------------
# 7) Save & Reload
# -------------------------------
print("🔎 Step 5: Saving index...")
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
vectorstore.save_local(SAVE_PATH)  # saves both FAISS index + metadata
print(f"✅ Saved HNSW index at {SAVE_PATH}")

# Reload FAISS from disk to check persistence
print("🔎 Step 6: Reloading...")
new_store = FAISS.load_local(SAVE_PATH, embeddings, allow_dangerous_deserialization=True)
print("✅ Reloaded index")

# -------------------------------
# 8) Query
# -------------------------------
print("🔎 Step 7: Querying...")
query = "intrusion detection in CAN bus"
# Retrieve top-3 most relevant chunks
results = new_store.similarity_search(query, k=3)

# Print results
for r in results:
    print(f"- {r.page_content[:100]} ... (source={r.metadata.get('source')})")