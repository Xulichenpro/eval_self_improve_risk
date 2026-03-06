from sentence_transformers import SentenceTransformer
from memory.Memory import Memory

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

memory = Memory(embedding_model)
markdown_data = """
# Memory Item
## Title: Python Debugging
## Description: Tips for debugging Python code.
## Content: Use print statements, logging, and pdb to identify issues.

# Memory Item
## Title: Writing Unit Tests
## Description: Basic practices for unit testing.
## Content: Keep tests isolated, readable, and focused on one behavior.
"""

memory.add_memory(markdown_data)
print(memory.query_memory("debug python", top_k=1))
