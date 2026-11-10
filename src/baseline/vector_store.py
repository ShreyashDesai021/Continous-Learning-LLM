"""Local Qdrant vector store wrapper supporting snapshot-specific collections."""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from src.core.models import DocumentClause, RetrievedChunk


class LocalQdrantStore:
    """Embedded Qdrant client persisting on D: drive or in-memory."""

    def __init__(
        self,
        storage_path: Optional[str] = "runtime/research/qdrant_storage",
        dimension: int = 1024,
        distance: Distance = Distance.COSINE
    ):
        self.dimension = dimension
        self.distance = distance
        
        if storage_path:
            p = Path(storage_path)
            p.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(p))
        else:
            self.client = QdrantClient(":memory:")

    def close(self) -> None:
        """Explicitly closes client to release file locks cleanly before shutdown."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def create_collection_if_not_exists(self, collection_name: str) -> None:
        """Creates collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=self.distance)
            )

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentClause],
        embeddings: np.ndarray
    ) -> None:
        """Indexes document chunks and their dense embeddings into Qdrant."""
        self.create_collection_if_not_exists(collection_name)
        
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.clause_id))
            payload = {
                "clause_id": chunk.clause_id,
                "document_id": chunk.document_id,
                "clause_number": chunk.clause_number,
                "text": chunk.text,
                "formatted_chunk": chunk.formatted_chunk,
                "metadata": chunk.metadata
            }
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        self.client.upsert(collection_name=collection_name, points=points)

    def search_filtered(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        query_filter: Optional[Filter] = None
    ) -> List[RetrievedChunk]:
        """Performs cosine similarity search against the snapshot collection with optional payload filter."""
        res = self.client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter
        )

        retrieved = []
        for hit in res.points:
            payload = hit.payload
            retrieved.append(RetrievedChunk(
                clause_id=payload["clause_id"],
                document_id=payload["document_id"],
                score=float(hit.score),
                text=payload["text"],
                metadata=payload.get("metadata", {})
            ))
        return retrieved

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Performs cosine similarity search against the snapshot collection."""
        return self.search_filtered(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            query_filter=None
        )
