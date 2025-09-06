# -------------------------------
# 1. Import loaders
# -------------------------------
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# -------------------------------
# 2. Initialize DirectoryLoader
# -------------------------------
# This will scan the "books" folder for all files matching *.pdf
# and use PyPDFLoader for each one.
loader = DirectoryLoader(
    path="/home/lisa/Arupreza/LangChain/Python/6_Document_Loaders/Files/",          # folder containing your PDF files
    glob="*.pdf",          # match only .pdf files
    loader_cls=PyPDFLoader # use PyPDFLoader for each file
)

# -------------------------------
# 3. Load all PDF files
# -------------------------------
# Option 1: Get a generator (lazy loading, one by one)
docs_generator = loader.lazy_load()

# Option 2: Load into a list of Document objects
docs = loader.load()

# -------------------------------
# 4. Inspect documents
# -------------------------------
# Iterate over the generator or list
for doc in docs:
    print(doc.metadata)   # metadata includes source path and page number
    # print(doc.page_content[:200])  # uncomment to see first 200 chars of text
