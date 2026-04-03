from enum import Enum

class VectorDBEnums(Enum):
    """
    Enum class for VectorDB related constants.
    """
    # VectorDB Types
    QDRANT = "QDRANT"
    PGVECTOR = "PGVECTOR"
    PINECONE = "PINECONE"
    WEAVIATE = "WEAVIATE"
    MILVUS = "MILVUS"


class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"

class PgVectorTablesSchemeEnums(Enum):
    ID= 'id'
    TEXT = 'text'
    VECTOR = 'vector'
    CHUNK_ID = 'chunk_id'
    METADATA = 'metadata'
    _PREFIX = 'pgvector'

class PgVectorDistanceMethodEnums(Enum):
    COSINE = "vector_cosine_ops"
    DOT = "vector_12_ops"

class PgVectorIndexTypeEnums(Enum):
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"