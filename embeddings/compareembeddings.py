
import ollama
import numpy as np


# Helper function to generate embeddings using local Ollama model
def get_embedding(text, model="all-minilm"):

    response = ollama.embed(
        model=model,
        input=text
    )

    return response["embeddings"][0]


# Calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)

    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    return dot_product / (magnitude1 * magnitude2)


# Compare different texts
texts = [
    "I love programming in Python",
    "Python coding is amazing",
    "I enjoy cooking Italian food",
    "Software development with Python"
]


# Calculate embeddings for each text
embeddings = [get_embedding(text) for text in texts]


# Compare similarities for all sentences
for i, text1 in enumerate(texts):

    for j, text2 in enumerate(texts):

        if i < j:  # Avoid duplicate comparisons

            similarity = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            print(f"'{text1}' vs '{text2}': {similarity:.3f}")
