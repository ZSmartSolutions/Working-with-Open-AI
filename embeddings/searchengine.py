import ollama
import numpy as np

from typing import List, Tuple


class SemanticSearch:

    def __init__(self):
        # Store documents and their corresponding embeddings
        self.documents = []
        self.embeddings = []

        # Local Ollama embedding model
        self.model = "all-minilm"

    def add_documents(self, documents: List[str]):
        """Add documents to the search index"""

        print(f"Adding {len(documents)} documents to index...")

        for doc in documents:

            self.documents.append(doc)

            # Generate embedding using Ollama
            embedding = self.get_embedding(doc)

            self.embeddings.append(embedding)

        print(f"Index now contains {len(self.documents)} documents")

    def get_embedding(self, text: str):
        """Generate embedding using local Ollama model"""

        response = ollama.embed(
            model=self.model,
            input=text
        )

        return response["embeddings"][0]

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for most relevant documents"""

        # Generate embedding for search query
        query_embedding = self.get_embedding(query)

        # Calculate similarities
        similarities = []

        for i, doc_embedding in enumerate(self.embeddings):

            similarity = cosine_similarity(
                query_embedding,
                doc_embedding
            )

            similarities.append((self.documents[i], similarity))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top matching documents
        return similarities[:top_k]


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)

    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


# Create and populate search engine
search_engine = SemanticSearch()


# Sample knowledge base
documents = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "Machine learning algorithms can identify patterns in large datasets automatically.",
    "React is a JavaScript library for building user interfaces, especially web applications.",
    "Artificial intelligence is revolutionizing industries from healthcare to transportation.",
    "Database management systems store and organize data efficiently for applications.",
    "Cloud computing provides on-demand access to computing resources over the internet.",
    "Cybersecurity protects digital systems from unauthorized access and cyber threats.",
    "Web development involves creating websites and web applications using various technologies."
]


# Add documents to search index
search_engine.add_documents(documents)


# Test different types of searches
queries = [
    "programming languages",
    "AI and ML",
    "building websites",
    "data storage",
    "online security"
]

# Returning tok  2 result
for query in queries:

    print(f"\nSearch: '{query}'")

    results = search_engine.search(query, top_k=2)

    for doc, similarity in results:

        print(f"  {similarity:.3f}: {doc}")