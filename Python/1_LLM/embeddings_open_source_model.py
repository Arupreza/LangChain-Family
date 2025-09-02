from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'BAAI/bge-base-en-v1.5')

result = embedding.embed_query("Dhaka is the capital of Bangladesh")

print(result)