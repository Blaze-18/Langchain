from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B"
)

document = [ 
    "Hello, My name is Anan",
    "I am in Love with Irina",
    "I had a pet who died"
]
text = "What is the embedding dimension of this model"
embedding1 = embeddings.embed_documents(document)
vector = embeddings.embed_query(text)


print(len(embedding1))
print(f"Embedding dimension = {len(vector)}")

# Sementic Search 

