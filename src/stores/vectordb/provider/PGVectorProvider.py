from ..VectorDBInterface import VectorDBInterface
from qdrant_client import QdrantClient,models
from models.db_schemes import RetrievedDocument
from ..VectorDBEnums import (DistanceMethodEnums ,PgVectorDistanceMethodEnums,
                              PgVectorIndexTypeEnums, PgVectorTablesSchemeEnums)
import logging
from typing import List
from sqlalchemy.sql import text as sql_text
from sqlalchemy.exc import IntegrityError
import json

class PGVectorProvider(VectorDBInterface):
    def __init__(self, db_client, default_vector_size: int=786,
                  distance_method: str=None, index_threshold: int=100):

        self.client = db_client

        if distance_method == DistanceMethodEnums.COSINE.value:
            distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT.value:
            distance_method = PgVectorDistanceMethodEnums.DOT.value
        
        self.default_vector_size = default_vector_size
        self.index_threshold = index_threshold
        self.pgvector_table_prefix = PgVectorTablesSchemeEnums._PREFIX.value
        self.distance_method = distance_method

        self.logger = logging.getLogger('uvicorn.error')
        self.default_index_name = lambda collection_name: f"{collection_name}_vector_idx"

    
    async def connect(self):
        async with self.client() as session:
            try:
                async with session.begin():
                    await session.execute(sql_text(
                        "CREATE EXTENSION IF NOT EXISTS vector"
                    ))

                await session.commit()
            except IntegrityError as exception:
                await session.rollback()
                if "pg_extension_name_index" not in str(exception):
                    raise
        
    async def disconnect(self):
        pass

    async def is_collection_existed(self, collection_name: str) -> bool:
        
        table_exists=None
        async with self.client() as session:
            async with session.begin():
                list_tbl = sql_text('SELECT * FROM pg_tables WHERE tablename = :collection_name')
                result = await session.execute(list_tbl, {"collection_name": collection_name})
                table_exists = result.scalar_one_or_none() is not None
        
        return table_exists
        
    async def list_all_collections(self) -> List[str]:

        tables = []
        async with self.client() as session:
            async with session.begin():
                list_tbl = sql_text('SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix')
                result = await session.execute(list_tbl, {"prefix": f"{self.pgvector_table_prefix}%"})
                tables = result.scalars().all()
        
        return tables
    
    async def get_collection_info(self, collection_name: str) -> dict:

        async with self.client() as session:
            async with session.begin():
                list_tbl = sql_text(f'''
                                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes 
                                    FROM pg_tables
                                    WHERE tablename = :collection_name
                                    ''')
                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')
                table_info = await session.execute(list_tbl, {"collection_name": collection_name})
                record_count = await session.execute(count_sql)
                
                table_info=table_info.fetchone()

                if not table_info:
                    return None
                
                return {
                    "table_info": {
                        "schemaname": table_info[0],
                        "tablename": table_info[1],
                        "tableowner": table_info[2],
                        "tablespace": table_info[3],
                        "hasindexes": table_info[4],
                    },
                    "record_count": record_count.scalar_one(),
                }

        
    async def delete_collection(self, collection_name: str):

        async with self.client() as session:
            async with session.begin():
                self.logger.info(f"Deleting collection '{collection_name}'...")
                drop_tbl = sql_text(f'DROP TABLE IF EXISTS {collection_name}')
                await session.execute(drop_tbl)
                await session.commit()
        return True

    async def create_collection(self, collection_name: str,
                                embedding_size: int=None,
                                do_reset: bool=False):
        if do_reset:
            _ = await self.delete_collection(collection_name)
        
        is_collection_existed = await self.is_collection_existed(collection_name)
        if is_collection_existed:
            return False

        self.logger.info(f"Creating collection '{collection_name}' with embedding size {embedding_size}...")
        async with self.client() as session:
            async with session.begin():
                create_tbl = sql_text(
                    f'CREATE TABLE {collection_name} ('
                        f'{PgVectorTablesSchemeEnums.ID.value} bigserial PRIMARY KEY,'
                        f'{PgVectorTablesSchemeEnums.TEXT.value} text, '
                        f'{PgVectorTablesSchemeEnums.VECTOR.value} vector({embedding_size}), '
                        f'{PgVectorTablesSchemeEnums.METADATA.value} jsonb DEFAULT \'{{}}\', '
                        f'{PgVectorTablesSchemeEnums.CHUNK_ID.value} integer, '
                        f'FOREIGN KEY ({PgVectorTablesSchemeEnums.CHUNK_ID.value}) REFERENCES data_chunks(chunk_id)'
                    ')'
                
                )
                await session.execute(create_tbl)
                await session.commit()
        
        return True
    
    async def is_index_existed(self, collection_name: str, index_name: str) -> bool:
        index_name = self.default_index_name(collection_name)

        async with self.client() as session:
            async with session.begin():
                lcheck_sql = sql_text("""
                    SELECT 1
                    FROM pg_indexes
                    WHERE tablename = :collection_name
                    AND indexname = :index_name
                    """)
                result = await session.execute(lcheck_sql, {"collection_name": collection_name, "index_name": index_name})
                index_exists = result.scalar_one_or_none() is not None
        return index_exists
    
    async def create_index(self, collection_name: str, index_type: str= PgVectorIndexTypeEnums.HNSW.value):
        
        is_index_existed = await self.is_index_existed(collection_name, self.default_index_name(collection_name))
        if is_index_existed:
            return False
                
        async with self.client() as session:
            async with session.begin():
                count_sql = sql_text(f'SELECT COUNT(*) FROM {collection_name}')
                result = await session.execute(count_sql)
                record_count = result.scalar_one()
            
                if record_count < self.index_threshold:
                    return False

                self.logger.info(f"START: Creating vector index for collection: {collection_name}")
                
                index_name = self.default_index_name(collection_name)
                creat_idx_sql = sql_text(f'CREATE INDEX {index_name} ON {collection_name} USING {index_type} ({PgVectorTablesSchemeEnums.VECTOR.value} {self.distance_method})')
                await session.execute(creat_idx_sql)
                await session.commit()
                self.logger.info(f"ENd: Creating vector index for collection: {collection_name}")

    async def reset_vector_index(self, collection_name: str, index_type: str= PgVectorIndexTypeEnums.HNSW.value):
        
        index_name = self.default_index_name(collection_name)
        async with self.client() as session:
            async with session.begin():
                drop_idx_sql = sql_text(f'DROP INDEX IF EXISTS {index_name}')
                await session.execute(drop_idx_sql)
                await session.commit()
        
        return await self.create_index(collection_name, index_type)

    async def insert_one(self, collection_name: str, text: str,vector:list,
                         metadata: dict = None, record_id: str = None):
        is_collection_existed = await self.is_collection_existed(collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection '{collection_name}' does not exist. Cannot insert record.")
            return False
        
        if not record_id:
            self.logger.warning("No record_id provided. Using auto-generated ID.")
            return False
        
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata is not None else "{}"
        
        async with self.client() as session:
            async with session.begin():
                insert_sql = sql_text(f'''
                    INSERT INTO {collection_name} ({PgVectorTablesSchemeEnums.TEXT.value}, {PgVectorTablesSchemeEnums.VECTOR.value}, {PgVectorTablesSchemeEnums.METADATA.value}, {PgVectorTablesSchemeEnums.CHUNK_ID.value})
                    VALUES (:text, :vector, :metadata, :chunk_id)
                ''')
                await session.execute(insert_sql, {
                    "text": text, 
                    "vector": "[" + ",".join([str(v) for v in vector]) + "]",  # Convert list to string format for SQL
                    "metadata": metadata_json, 
                    "chunk_id": record_id
                                                   })
                
                await session.commit()
                await self.create_index(collection_name=collection_name)
        
        return True
    
    async def insert_many(self, collection_name: str, texts: List[str], vectors: List[list],
                          metadatas: List[dict] = None, record_ids: List[str] = None, batch_size: int = 50):
        
        is_collection_existed = await self.is_collection_existed(collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection '{collection_name}' does not exist. Cannot insert records.")
            return False
        
        if batch_size <= 0:
            self.logger.error("batch_size must be greater than zero.")
            return False

        if len(texts) != len(vectors):
            self.logger.error("Length of texts and vectors must be the same.")
            return False

        if record_ids is not None and len(record_ids) != len(texts):
            self.logger.error("Length of record_ids must match texts length.")
            return False

        if metadatas is not None and len(metadatas) not in (0, len(texts)):
            self.logger.error("Length of metadatas must be zero or match texts length.")
            return False
        
        if not metadatas or len(metadatas) == 0:
            metadatas = [None] * len(texts)

        async with self.client() as session:
            async with session.begin():
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_vectors = vectors[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size] if metadatas else [None] * len(batch_texts)
                    batch_record_ids = record_ids[i:i+batch_size] if record_ids else [None] * len(batch_texts)

                    values=[]
                    for text, vector, metadata, record_id in zip(batch_texts, batch_vectors, batch_metadatas, batch_record_ids):
                        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata is not None else "{}"
                        values.append({
                            "text": text, 
                            "vector": "[" + ",".join([str(v) for v in vector]) + "]",  # Convert list to string format for SQL
                            "metadata": metadata_json, 
                            "chunk_id": record_id
                        })

                    batch_insert_sql = sql_text(f'''
                            INSERT INTO {collection_name} ({PgVectorTablesSchemeEnums.TEXT.value}, {PgVectorTablesSchemeEnums.VECTOR.value}, {PgVectorTablesSchemeEnums.METADATA.value}, {PgVectorTablesSchemeEnums.CHUNK_ID.value})
                            VALUES (:text, :vector, :metadata, :chunk_id)
                        ''')
                    if values:
                        await session.execute(batch_insert_sql, values)
                    
                await session.commit()

        await self.create_index(collection_name=collection_name)
                
        return True
    
    async def search_by_vector(self, collection_name: str, 
                         vector: list, limit: int):
        
        is_collection_existed = await self.is_collection_existed(collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection '{collection_name}' does not exist. Cannot perform search.")
            return []
        
        vector = "[" + ",".join([str(v) for v in vector]) + "]"  # Convert list to string format for SQL

        async with self.client() as session:
            async with session.begin():
                search_sql = sql_text(f'''
                    SELECT 
                        {PgVectorTablesSchemeEnums.TEXT.value} as text, 
                        {PgVectorTablesSchemeEnums.CHUNK_ID.value} as chunk_id,
                        {PgVectorTablesSchemeEnums.METADATA.value} as metadata,
                        1 - ({PgVectorTablesSchemeEnums.VECTOR.value} <=> :vector) as score
                    FROM {collection_name}
                    ORDER BY score DESC
                    LIMIT {limit}
                ''')
                result = await session.execute(search_sql, {"vector": vector})
                records = result.fetchall()

                return [
                    RetrievedDocument(
                        text=record.text, 
                        score=record.score,
                        chunk_id=record.chunk_id,
                        metadata=record.metadata
                        ) 
                        for record in records
                ]
        
            

        
    