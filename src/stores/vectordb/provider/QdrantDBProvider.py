from ..VectorDBInterface import VectorDBInterface
from qdrant_client import QdrantClient,models
from models.db_schemes import RetrievedDocument
from ..VectorDBEnums import DistanceMethodEnums
import logging

class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_path:str, distance_method:str):

        self.client = None
        self.db_path = db_path
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.EUCLIDEAN.value:
            self.distance_method = models.Distance.EUCLIDEAN
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT
        
        self.logger = logging.getLogger(__name__)

    
    def connect(self):
        try:
            self.client = QdrantClient(path=self.db_path)
            self.logger.info("Connected to QdrantDB successfully.")
        except Exception as e:
            self.logger.error(f"Failed to connect to QdrantDB: {e}")
            self.client = None
        
    def disconnect(self):
        self.client = None
        self.logger.info("Disconnected from QdrantDB.")
    
    def is_collection_existed(self, collection_name: str) -> bool:
        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return False
        try:
            return self.client.collection_exists(collection_name=collection_name)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False
    
    def list_all_collections(self) -> list:
        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return []
        try:
            collections = self.client.get_collections()
            return collections
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> dict:
        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return {}
        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
            return collection_info
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self, collection_name: str):
        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return
        
        if self.is_collection_existed(collection_name):
            try:
                self.client.delete_collection(collection_name=collection_name)
                self.logger.info(f"Collection '{collection_name}' deleted successfully.")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting collection: {e}")
                return False
            
    def create_collection(self, collection_name: str,
                           embedding_size: int,
                           do_reset: bool = False
                           ):
        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return
        
        if self.is_collection_existed(collection_name):
            if do_reset:
                self.delete_collection(collection_name)
            else:
                self.logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
                return
        
        try:
            _ = self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )
            self.logger.info(f"Collection '{collection_name}' created successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False
        
    def insert_one(self, collection_name: str, text: str,vector:list,
                    metadata: dict = None,
                    record_id: str = None
                        ):
        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist. Please create it before inserting data.")
            return
        
        try:
            self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=record_id,
                        vector=vector,
                        payload={
                            "text": text,
                            "metadata": metadata
                        }
                    )
                ]
            )
            self.logger.info(f"Inserted one record into collection '{collection_name}' successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error inserting record: {e}")
            return False
    
    def insert_many(self, collection_name: str, texts: list, vectors: list,
                    metadatas: list = None,
                    record_ids: list = None,
                    batch_size: int = 50
                    ):
        if metadatas is None:
            metadatas = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist. Please create it before inserting data.")
            return
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_end= i + batch_size
                batch_texts = texts[i:batch_end]
                batch_vectors = vectors[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                batch_record_ids = record_ids[i:batch_end]

                batch_records= [
                    models.Record(
                        id=batch_record_ids[j],
                        vector=batch_vectors[j],
                        payload={
                            "text": batch_texts[j],
                            "metadata": batch_metadatas[j]
                        }
                    ) for j in range(len(batch_texts))
                ]

                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records
                )
            self.logger.info(f"Inserted {len(texts)} records into collection '{collection_name}' successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error inserting many records: {e}")
            return False
        
    def search_by_vector(self, collection_name: str, 
                         vector: list, limit: int=5):
        if not self.client:
            self.logger.error("Qdrant client is not initialized.")
            return []
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection '{collection_name}' does not exist.")
            return []
        
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit
            )
            if not search_result or len(search_result) == 0:
                self.logger.info(f"No search results found for the given vector in collection '{collection_name}'.")
                return []
            
            return [
                RetrievedDocument(**{
                    "text": record.payload.get("text", ""),
                    "score": record.score
                }) for record in search_result
            ]
        except Exception as e:
            self.logger.error(f"Error searching by vector: {e}")
            return []