import re
from typing import Any, Dict, List, Tuple


class Memory:
    """
    A simple memory store based on title embeddings.

    Attributes:
        memories: A list of (embed, text) tuples.
                  - embed: vector returned by embedding_model
                  - text: dict with keys: title, description, content
        embedding_model: embedding model object used to encode text
    """

    def __init__(self, embedding_model: Any):
        """
        Initialize Memory.

        Args:
            embedding_model: An object that can encode text into vectors.
                             It should provide either:
                             - encode(text)
                             - get_embedding(text)
        """
        self.memories: List[Tuple[List[float], Dict[str, str]]] = []
        self.embedding_model = embedding_model

    def _encode(self, text: str):
        """Encode text using the provided embedding model."""
        if hasattr(self.embedding_model, "encode"):
            return self.embedding_model.encode(text)
        if hasattr(self.embedding_model, "get_embedding"):
            return self.embedding_model.get_embedding(text)
        raise AttributeError(
            "embedding_model must provide an 'encode' or 'get_embedding' method."
        )

    @staticmethod
    def _to_float_list(vec) -> List[float]:
        """Convert a vector-like object into a Python float list."""
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        return [float(x) for x in vec]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length.")

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def add_memory(self, markdown_text: str) -> None:
        """
        Parse multiple memory items from a markdown string and add them to memories.

        Expected item format:
            # Memory Item
            ## Title: <title>
            ## Description: <description>
            ## Content: <content>

        Args:
            markdown_text: A string containing one or more memory items.
        """
        pattern = re.compile(
            r"#\s*Memory\s*Item\s*"
            r"##\s*Title:\s*(.*?)\s*"
            r"##\s*Description:\s*(.*?)\s*"
            r"##\s*Content:\s*(.*?)(?=\n#\s*Memory\s*Item\s*|\Z)",
            re.DOTALL | re.IGNORECASE,
        )

        matches = pattern.findall(markdown_text)
        for title, description, content in matches:
            title = title.strip()
            description = description.strip()
            content = content.strip()

            embed = self._to_float_list(self._encode(title))
            text = {
                "title": title,
                "description": description,
                "content": content,
            }
            self.memories.append((embed, text))

    def query_memory(self, query: str, top_k: int = 3) -> str:
        """
        Query the memory store by cosine similarity.

        Args:
            query: Query string.
            top_k: Number of top memory items to return. Default is 3.

        Returns:
            A markdown string containing the top_k matched memory items.
        """
        if not self.memories:
            return ""

        embed_query = self._to_float_list(self._encode(query))

        scored = []
        for embed, text in self.memories:
            score = self._cosine_similarity(embed, embed_query)
            scored.append((score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = scored[:max(0, top_k)]

        result_blocks = []
        for _, text in top_items:
            block = (
                "# Memory Item\n"
                f"## Title: {text['title']}\n"
                f"## Description: {text['description']}\n"
                f"## Content: {text['content']}"
            )
            result_blocks.append(block)

        return "\n\n".join(result_blocks)