from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Load your .env file so the Hugging Face token is available
load_dotenv()

# Make sure you set your token in .env:
# HUGGINGFACEHUB_API_TOKEN=hf_xxx

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    task="text-generation",
    max_new_tokens=128,
    temperature=0.2,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    timeout=120
)

model = ChatHuggingFace(llm=llm)

# Send a message
result = model.invoke([HumanMessage(content="What is the capital of Bangladesh?")])

print(result.content)
