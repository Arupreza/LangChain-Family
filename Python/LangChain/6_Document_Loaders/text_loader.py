# -------------------------------
# 1. Import required libraries
# -------------------------------
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables (e.g., Hugging Face API key if stored in .env)
load_dotenv()

# -------------------------------
# 2. Define model IDs
# -------------------------------
gemma_model = "google/gemma-2-2b-it"
llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -------------------------------
# 3. Initialize Hugging Face model pipeline
# -------------------------------
# Using Gemma as the generator
model = HuggingFacePipeline.from_model_id(
    model_id=gemma_model,
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,       # controls randomness (lower = more deterministic)
        max_new_tokens=100,    # maximum new tokens to generate
        return_full_text=False # only return generated text, not input + output
    )
)

# -------------------------------
# 4. Define prompt for summarization
# -------------------------------
# This will take the poem text and ask the LLM to summarize it
prompt = PromptTemplate(
    template='Write a summary for the following poem:\n\n{poem}',
    input_variables=['poem']
)

# Parser that converts raw LLM output into plain string
parser = StrOutputParser()

# -------------------------------
# 5. Load text document safely
# -------------------------------
# Build absolute path so script works regardless of where you run it from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Files", "cricket.txt")

# Load file with explicit UTF-8 encoding (fixes RuntimeError from loader)
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# -------------------------------
# 6. Inspect loaded docs
# -------------------------------
print(type(docs))               # should be a list of Document objects
print(len(docs))                # number of documents (usually 1 for .txt)
print(docs[0].page_content[:200])  # print first 200 characters of content
print(docs[0].metadata)         # shows file source path

# -------------------------------
# 7. Build the chain
# -------------------------------
# Flow: poem text → prompt → model → parser
chain = prompt | model | parser

# -------------------------------
# 8. Run the chain
# -------------------------------
# Pass the poem text into the chain for summarization
print(chain.invoke({'poem': docs[0].page_content}))