import ollama

response = ollama.embed(
    model="all-minilm",
    input="What is Retrieval Augmented Generation?"
)

embedding = response["embeddings"][0]

print("Dimensions:", len(embedding))
print("Vector:", embedding[:10])