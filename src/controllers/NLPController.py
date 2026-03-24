from .BaseController import BaseController
from models.db_schemes import project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnums
from typing import List
import json
class NLPController(BaseController):
    
    def __init__(self, vectordb_client, generation_client, embedding_client):
        super().__init__()
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client

    
    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    

    def reset_vector_db_collection(self, project: project):
        collection_name = self.create_collection_name(project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: project):
        collection_name = self.create_collection_name(project.project_id)
        return self.vectordb_client.get_collection_info(collection_name=collection_name)
    
    def index_into_vector_db(self, project:project, chunks:List[DataChunk],
                             chunks_ids: List[int],
                             do_reset: bool = False):
        
        collection_name = self.create_collection_name(project.project_id)

        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]

        vectors=[
            self.embedding_client.embed_text(text=text, 
                                             document_type=DocumentTypeEnums.DOCUMENT.value)
            for text in texts
        ]
        _ = self.vectordb_client.create_collection(collection_name=collection_name,
                                                  embedding_size=self.embedding_client.embedding_size,
                                                    do_reset=do_reset
                                                  )
        _ = self.vectordb_client.insert_many(collection_name=collection_name,
                                              texts=texts, 
                                              vectors=vectors, 
                                              metadatas=metadatas,
                                              record_ids=chunks_ids
                                              )
        
        return True
    
    def search_vector_db_collection(self, project:project, text:str, limit:int=10):

        collection_name = self.create_collection_name(project.project_id)

        vector= self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnums.QUERY.value)

        if not vector or len(vector) == 0:
            self.logger.error(f"Failed to get embedding vector for text: {text}")
            return []
        
        results = self.vectordb_client.search_by_vector(collection_name=collection_name,
                                                         vector=vector,
                                                           limit=limit)
        if results is None:
            self.logger.error(f"Failed to search vector database for project_id: {project.project_id}, text: {text}")
            return []
        results= json.loads(
            json.dumps(results, default=lambda x: x.__dict__)
        )
        return results