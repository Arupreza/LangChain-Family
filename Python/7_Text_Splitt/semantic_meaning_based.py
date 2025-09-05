# -------------------------------
# 1. Import required classes
# -------------------------------
# SemanticChunker is in langchain_experimental
from langchain_experimental.text_splitter import SemanticChunker

# OpenAI embeddings (make sure you set OPENAI_API_KEY in your .env)
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

# -------------------------------
# 2. Load environment variables
# -------------------------------
# This loads OPENAI_API_KEY from your .env file
load_dotenv()

# -------------------------------
# 3. Initialize SemanticChunker
# -------------------------------
# - OpenAIEmbeddings() → used to calculate semantic similarity
# - breakpoint_threshold_type="standard_deviation" → how to decide splits
# - breakpoint_threshold_amount=3 → stricter threshold (fewer chunks)
text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

# -------------------------------
# 4. Define sample text
# -------------------------------
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# -------------------------------
# 5. Split into semantic chunks
# -------------------------------
# create_documents() returns a list of Document objects
docs = text_splitter.create_documents([sample])

# -------------------------------
# 6. Inspect results
# -------------------------------
print(f"Number of semantic chunks: {len(docs)}")

for i, doc in enumerate(docs, 1):
    print(f"\n--- Chunk {i} ---")
    print(doc.page_content)