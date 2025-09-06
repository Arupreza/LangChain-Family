# -------------------------------
# 1. Import required classes
# -------------------------------
from langchain_text_splitters import CharacterTextSplitter   # ✅ new import path in latest LangChain
from langchain_community.document_loaders import PyPDFLoader

# -------------------------------
# 2. Load the PDF
# -------------------------------
# PyPDFLoader extracts text page by page from your PDF file.
loader = PyPDFLoader("/home/lisa/Arupreza/LangChain/Python/7_Text_Splitt/dl-curriculum.pdf")

# Each page will be returned as a Document object
docs = loader.load()

print(f"Total pages loaded: {len(docs)}")   # check number of pages in the PDF

# -------------------------------
# 3. Initialize text splitter
# -------------------------------
# CharacterTextSplitter breaks text into smaller chunks.
# - chunk_size = 200 characters
# - chunk_overlap = 0 (no overlap between chunks)
# - separator = "\n" (split on newlines, default is recommended instead of '')
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator="\n"   # ✅ safer than '' (empty string)
)

# -------------------------------
# 4. Split the documents
# -------------------------------
# Each page (Document) will be broken into smaller Document chunks.
result = splitter.split_documents(docs)

# -------------------------------
# 5. Inspect results
# -------------------------------
print(f"Total chunks created: {len(result)}")
print(result[0])   # preview first chunk (Document object with text + metadata)
print(result[0].page_content)  # just the text of the first chunk
print(result[0].metadata)      # metadata like source file and page number