from enum import Enum

class VectorDBEnums(Enum):
    """
    Enum class for VectorDB related constants.
    """
    # VectorDB Types
    QDRANT = "QDRANT"
    PINECONE = "PINECONE"
    WEAVIATE = "WEAVIATE"
    MILVUS = "MILVUS"


class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"