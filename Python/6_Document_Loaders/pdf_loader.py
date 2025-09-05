# -------------------------------
# 1. Import PDFPlumberLoader
# -------------------------------
from langchain_community.document_loaders import PDFPlumberLoader
import os

# -------------------------------
# 2. Build absolute path to PDF
# -------------------------------
# This ensures the script works no matter where it is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Files", "dl-curriculum.pdf")

# -------------------------------
# 3. Initialize loader with your PDF file
# -------------------------------
# PDFPlumberLoader uses the pdfplumber library to extract text.
# Unlike TextLoader, it does NOT need an encoding argument.
loader = PDFPlumberLoader(file_path)

# -------------------------------
# 4. Load the PDF into Document objects
# -------------------------------
docs = loader.load()

# -------------------------------
# 5. Inspect results
# -------------------------------
print(len(docs))                 # number of Document objects (usually = number of pages)
print(docs[0].page_content)      # text content of the first page
print(docs[1].metadata)          # metadata (source file path, page number, etc.)