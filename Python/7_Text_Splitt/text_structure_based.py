# -------------------------------
# 1. Import RecursiveCharacterTextSplitter
# -------------------------------
# In new versions of LangChain, text splitters are in a separate package.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------
# 2. Define input text
# -------------------------------
# This is the raw text we want to split into chunks.
text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# -------------------------------
# 3. Initialize the splitter
# -------------------------------
# RecursiveCharacterTextSplitter is smarter than CharacterTextSplitter:
# - It tries to split by paragraphs -> sentences -> words -> characters
# - chunk_size = max number of characters in each chunk
# - chunk_overlap = overlap between chunks to preserve context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,  # max size of each chunk
    chunk_overlap=25 # number of overlapping characters between chunks
)

# -------------------------------
# 4. Perform the split
# -------------------------------
# For plain text, use split_text()
chunks = splitter.split_text(text)

# -------------------------------
# 5. Inspect results
# -------------------------------
print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)
