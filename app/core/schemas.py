from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, constr, conlist

# --- Embedding Schemas ---

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        ..., 
        description="Input text or list of texts to embed",
        min_length=1,
        examples=["Hello world", ["Hello", "World"]]
    )
    model: Optional[str] = Field(
        None, 
        description="The ID of the model to use"
    )

class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Optional[dict] = Field(default=None, description="Usage statistics")

# --- Reranker Schemas ---

class RerankRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=1, 
        description="The query to rerank passages against"
    )
    passages: List[str] = Field(
        ..., 
        min_items=1, 
        description="The list of passages to rerank"
    )
    model: Optional[str] = Field(
        None, 
        description="The ID of the model to use"
    )
    return_documents: bool = Field(
        True, 
        description="Whether to return the document text in the response"
    )

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None

class RerankResponse(BaseModel):
    object: str = "list"
    data: List[RerankResult]
    model: str
    usage: Optional[dict] = Field(default=None, description="Usage statistics")
