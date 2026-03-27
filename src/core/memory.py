import os
import uuid
import time
import math
import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

logger = logging.getLogger(__name__)

class LongTermMemory:
    """LTM Qdrant vector memory for strategy caching."""
    def __init__(self, collection_name: str = "naminsight_memory", url: str = "http://localhost:6333"):
        self.collection_name = collection_name
        self.url = os.getenv("QDRANT_URL", url)
        self.client = QdrantClient(url=self.url)
        if TextEmbedding is None:
            raise ImportError("fastembed is required for internal memory embeddings")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
    def ensure_collection(self):
        exists = self.client.collection_exists(self.collection_name)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def store_memory(self, task_description: str, successful_plan: str, score: float = 1.0):
        """Save vector memory of successful plans."""
        self.ensure_collection()
        
        timestamp = time.time()
        embedding = list(self.embedding_model.embed([task_description]))[0]
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "task": task_description,
                "plan": successful_plan,
                "score": score,
                "timestamp": timestamp
            }
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        logger.info("Memory successfully encoded and stored in Qdrant.")

    def retrieve_memory(self, task_description: str, limit: int = 1) -> Optional[dict]:
        """Retrieves past effective plans for a similar task."""
        self.ensure_collection()
        embedding = list(self.embedding_model.embed([task_description]))[0]
        
        # fetch using Qdrant v1.10+ query_points API
        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=limit,
            with_payload=True,
            score_threshold=0.85 
        )
        
        # unpack matching points
        if search_response.points:
            best_match = search_response.points[0]
            logger.info(f"Memory triggered! Found similar previous task (Score: {best_match.score})")
            return {"plan": best_match.payload.get("plan"), "score": best_match.score}
        return None

    def prune_memory(self, decay_rate: float = 0.1, threshold: float = 0.2):
        """Prune decayed memories: W = Score * e^(-lambda * t) < threshold"""
        self.ensure_collection()
        scroll_results = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True
        )
        points = scroll_results[0]
        current_time = time.time()
        to_delete = []
        
        for point in points:
            payload = point.payload or {}
            score = payload.get("score", 1.0)
            timestamp = payload.get("timestamp", current_time)
            
            days_elapsed = (current_time - timestamp) / (60 * 60 * 24)
            weight = score * math.exp(-decay_rate * days_elapsed)
            
            if weight < threshold:
                to_delete.append(point.id)
                logger.debug(f"Memory {point.id} pruned. Weight {weight:.3f} < {threshold}")
                
        if to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=to_delete
            )
            logger.info(f"Pruned {len(to_delete)} decayed memories from Qdrant.")
