import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

# typed IR schemas
class RAGInput(BaseModel):
    query: str = Field(..., description="The semantic search query applied to the RAG database.")
    limit: int = Field(5, description="Number of results to retrieve.")

class RAGCitation(BaseModel):
    source: str = Field(description="Origin source descriptor for the document snippet.")
    content: str = Field(description="The plain text chunk payload.")
    score: float = Field(description="Similarity score bridging query and the retrieved chunk.")

class RAGOutput(BaseModel):
    context: List[RAGCitation] = Field(description="List of retrieved context snippets and their citations.")

class RAGTool:
    """Async fastembed Qdrant RAG engine."""
    def __init__(self, collection_name: str = "naminsight_docs", url: str = "http://localhost:6333"):
        self.collection_name = collection_name
        # fastembed hooks synchronous ops inside async wrappers
        self.client = AsyncQdrantClient(url=url)
        # load fastembed auto-chunker
        self.client.set_model("BAAI/bge-small-en-v1.5")
        
    async def _ensure_collection(self):
        """Init DB collection"""
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            logger.info(f"Collection {self.collection_name} does not exist. It will be implicitly created upon insertion.")

    async def ingest_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Ingest pre-chunked doc strings."""
        await self._ensure_collection()
        logger.info(f"Ingesting {len(documents)} document chunks into {self.collection_name}...")
        
        # sync insert + auto-embed
        await self.client.add(
            collection_name=self.collection_name,
            documents=documents,
            metadata=metadata
        )
        logger.info("Ingestion strictly completed.")

    async def retrieve(self, query: str, limit: int = 5) -> RAGOutput:
        """Get semantic matches with citations."""
        await self._ensure_collection()
        logger.info(f"Retrieving knowledge for query: '{query}'")
        
        # auto-embed query text & search
        search_results = await self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=limit
        )
        
        citations = []
        for result in search_results:
            source = result.metadata.get("source", "Unknown Source") if result.metadata else "Unknown Source"
            content = result.document or ""
            score = result.score
            
            citations.append(RAGCitation(
                source=source,
                content=content,
                score=score
            ))
            
        return RAGOutput(context=citations)

    async def run(self, **kwargs) -> dict:
        """Tool runner entrypoint."""
        req = RAGInput(**kwargs)
        out = await self.retrieve(query=req.query, limit=req.limit)
        return out.model_dump()
