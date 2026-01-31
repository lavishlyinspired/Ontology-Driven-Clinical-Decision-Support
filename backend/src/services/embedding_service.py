"""
Embedding Service for Semantic Search
Provides vector embeddings for clinical terms, patient data, and guidelines.
Supports semantic similarity search across the knowledge base.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Result of semantic search"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source: str


class EmbeddingService:
    """
    Service for generating and managing text embeddings for semantic search.
    Supports multiple embedding providers and caching.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        cache_enabled: bool = True
    ):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, List[float]] = {}
        self._index: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

        # Pre-built clinical embeddings for common terms
        self._clinical_embeddings = self._initialize_clinical_embeddings()

    def _initialize_clinical_embeddings(self) -> Dict[str, List[float]]:
        """
        Initialize pre-computed embeddings for common clinical terms.
        In production, these would be loaded from a vector database.
        """
        # Simulated embeddings using deterministic hash-based vectors
        clinical_terms = [
            "non-small cell lung cancer",
            "small cell lung cancer",
            "adenocarcinoma",
            "squamous cell carcinoma",
            "egfr mutation",
            "alk rearrangement",
            "pd-l1 expression",
            "kras g12c mutation",
            "osimertinib",
            "pembrolizumab",
            "carboplatin pemetrexed",
            "stage iv",
            "metastatic disease",
            "brain metastasis",
            "first line treatment",
            "progression free survival",
            "overall survival",
            "complete response",
            "partial response",
            "stable disease"
        ]

        embeddings = {}
        for term in clinical_terms:
            embeddings[term.lower()] = self._generate_deterministic_embedding(term)

        return embeddings

    def _generate_deterministic_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic embedding based on text hash.
        Used for demo/testing - production would use actual embedding API.
        """
        # Create hash of text
        text_hash = hashlib.sha256(text.lower().encode()).hexdigest()

        # Convert hash to normalized vector
        np.random.seed(int(text_hash[:8], 16))
        embedding = np.random.randn(self.dimensions)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    async def generate_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate embedding for given text.

        Args:
            text: Text to embed
            metadata: Optional metadata to associate with embedding

        Returns:
            EmbeddingResult with embedding vector
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Cache hit for embedding: {text[:50]}...")
            return EmbeddingResult(
                text=text,
                embedding=self._cache[cache_key],
                model=self.model,
                dimensions=self.dimensions,
                metadata=metadata
            )

        # Check pre-built clinical embeddings
        text_lower = text.lower().strip()
        if text_lower in self._clinical_embeddings:
            embedding = self._clinical_embeddings[text_lower]
        else:
            # Generate embedding
            embedding = await self._call_embedding_api(text)

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = embedding

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model,
            dimensions=self.dimensions,
            metadata=metadata
        )

    async def _call_embedding_api(self, text: str) -> List[float]:
        """
        Call embedding API (OpenAI, Cohere, etc.)
        In production, this would make actual API calls.
        """
        try:
            if self.provider == "openai":
                return await self._openai_embedding(text)
            elif self.provider == "cohere":
                return await self._cohere_embedding(text)
            else:
                # Fallback to deterministic embedding
                return self._generate_deterministic_embedding(text)
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}, using fallback")
            return self._generate_deterministic_embedding(text)

    async def _openai_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using OpenAI API.
        Requires OPENAI_API_KEY environment variable.
        """
        try:
            import openai
            client = openai.AsyncOpenAI()
            response = await client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except ImportError:
            logger.warning("OpenAI library not installed, using fallback")
            return self._generate_deterministic_embedding(text)
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")
            return self._generate_deterministic_embedding(text)

    async def _cohere_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Cohere API.
        """
        try:
            import cohere
            client = cohere.AsyncClient()
            response = await client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]
        except ImportError:
            logger.warning("Cohere library not installed, using fallback")
            return self._generate_deterministic_embedding(text)
        except Exception as e:
            logger.warning(f"Cohere embedding failed: {e}")
            return self._generate_deterministic_embedding(text)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.model}:{text.lower().strip()}".encode()).hexdigest()

    async def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Index a document for semantic search.

        Args:
            doc_id: Unique document identifier
            text: Document text to embed
            metadata: Document metadata (source, type, etc.)
        """
        result = await self.generate_embedding(text, metadata)
        self._index[doc_id] = (result.embedding, {
            "text": text,
            "metadata": metadata
        })
        logger.info(f"Indexed document: {doc_id}")

    async def index_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Index multiple documents.

        Args:
            documents: List of dicts with 'id', 'text', 'metadata' keys

        Returns:
            Number of documents indexed
        """
        count = 0
        for doc in documents:
            await self.index_document(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata", {})
            )
            count += 1
        return count

    async def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            query: Search query text
            top_k: Maximum number of results
            min_score: Minimum similarity score (0-1)
            filters: Optional metadata filters

        Returns:
            List of SearchResult sorted by relevance
        """
        if not self._index:
            logger.warning("Search index is empty")
            return []

        # Generate query embedding
        query_result = await self.generate_embedding(query)
        query_embedding = np.array(query_result.embedding)

        results = []
        for doc_id, (doc_embedding, doc_data) in self._index.items():
            # Calculate cosine similarity
            doc_emb_array = np.array(doc_embedding)
            score = float(np.dot(query_embedding, doc_emb_array))

            # Apply score threshold
            if score < min_score:
                continue

            # Apply filters
            if filters:
                metadata = doc_data.get("metadata", {})
                if not self._matches_filters(metadata, filters):
                    continue

            results.append(SearchResult(
                id=doc_id,
                text=doc_data["text"],
                score=score,
                metadata=doc_data.get("metadata", {}),
                source=doc_data.get("metadata", {}).get("source", "unknown")
            ))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    async def find_similar(
        self,
        doc_id: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document.

        Args:
            doc_id: Document ID to find similar documents for
            top_k: Maximum number of results
            exclude_self: Whether to exclude the source document

        Returns:
            List of similar SearchResults
        """
        if doc_id not in self._index:
            logger.warning(f"Document not found in index: {doc_id}")
            return []

        doc_embedding, doc_data = self._index[doc_id]
        query_embedding = np.array(doc_embedding)

        results = []
        for other_id, (other_embedding, other_data) in self._index.items():
            if exclude_self and other_id == doc_id:
                continue

            other_emb_array = np.array(other_embedding)
            score = float(np.dot(query_embedding, other_emb_array))

            results.append(SearchResult(
                id=other_id,
                text=other_data["text"],
                score=score,
                metadata=other_data.get("metadata", {}),
                source=other_data.get("metadata", {}).get("source", "unknown")
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        sources = {}
        for doc_id, (_, data) in self._index.items():
            source = data.get("metadata", {}).get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1

        return {
            "total_documents": len(self._index),
            "cache_size": len(self._cache),
            "model": self.model,
            "dimensions": self.dimensions,
            "sources": sources
        }

    def clear_index(self) -> None:
        """Clear the search index"""
        self._index.clear()
        logger.info("Search index cleared")

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


async def semantic_search(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[SearchResult]:
    """
    Convenience function for semantic search.

    Args:
        query: Search query
        top_k: Maximum results
        filters: Optional metadata filters

    Returns:
        List of search results
    """
    service = get_embedding_service()
    return await service.search(query, top_k=top_k, filters=filters)


async def index_clinical_content(
    content_type: str,
    items: List[Dict[str, Any]]
) -> int:
    """
    Index clinical content (guidelines, treatments, etc.)

    Args:
        content_type: Type of content (guideline, treatment, biomarker, etc.)
        items: List of items to index

    Returns:
        Number of items indexed
    """
    service = get_embedding_service()

    documents = []
    for item in items:
        doc = {
            "id": f"{content_type}:{item.get('id', item.get('name', 'unknown'))}",
            "text": item.get("text", item.get("description", str(item))),
            "metadata": {
                "source": content_type,
                "type": content_type,
                **item
            }
        }
        documents.append(doc)

    return await service.index_batch(documents)
