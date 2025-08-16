from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B"
)

document = [
    "Anan likes to have his coffee black and strong",
    "Irina likes his coffee sweet and milky",
    "Nirjhor likes his coffee with suger cubes and little milk",
    "Rafsun don't like coffee that much but he likes chocolate iced coffee",
    "Enan only drinks tea no coffee"
]

query = "Tell me about the person who likes tea only"

doc_embedding = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(query)

print(f"Dimension of query: {len(query_embedding)}")

sim_scores = cosine_similarity([query_embedding], doc_embedding)[0]
sorted_scores = sorted(list(enumerate(sim_scores)), key=lambda x:x[1])
index, score = sorted_scores[-1]
print(f"sorted similiarity scores: {sim_scores}")
print(f"Highest Similiarity score: {score}")
print(f"Query: {query}")
print(f"Similiarity with the doc: {document[index]}\nScore: {score:.4f}")
