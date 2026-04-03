from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnums
from typing import List
import json
class NLPController(BaseController):
    
    def __init__(self, vectordb_client, generation_client, embedding_client, template_parser=None):
        super().__init__()
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
    
    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    

    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)
    
    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project.project_id)
        return await self.vectordb_client.get_collection_info(collection_name=collection_name)
    
    async def index_into_vector_db(self, project:Project, chunks:List[DataChunk],
                             chunks_ids: List[int],
                             do_reset: bool = False):
        
        collection_name = self.create_collection_name(project.project_id)

        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]

        vectors = self.embedding_client.embed_text(text=texts, 
                                             document_type=DocumentTypeEnums.DOCUMENT.value)
           
        _ = await self.vectordb_client.create_collection(collection_name=collection_name,
                                                  embedding_size=self.embedding_client.embedding_size,
                                                    
                                                  )
        _ = await self.vectordb_client.insert_many(collection_name=collection_name,
                                              texts=texts, 
                                              vectors=vectors, 
                                              metadatas=metadatas,
                                              record_ids=chunks_ids
                                              )
        
        return True
    
    async def search_vector_db_collection(self, project:Project, text:str, limit:int=10):

        collection_name = self.create_collection_name(project.project_id)

        vector= self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnums.QUERY.value)

        if not vector or len(vector) == 0:
            self.logger.error(f"Failed to get embedding vector for text: {text}")
            return []
        
        if isinstance(vector,list) and len(vector)>0:
            vector=vector[0]

        if not vector:
            return False
        
        results = await self.vectordb_client.search_by_vector(collection_name=collection_name,
                                                         vector=vector,
                                                           limit=limit)
        if results is None:
            self.logger.error(f"Failed to search vector database for project_id: {project.project_id}, text: {text}")
            return []
        
        return results
    

    async def answer_rag_question(self, project:Project, query:str, limit:int=10):
        answer,full_prompt, chat_history=None, None, None
        # step 1: search relevant documents in vector database
        retrieved_document= await self.search_vector_db_collection(project=project, text=query, limit=limit)

        if not retrieved_document or len(retrieved_document) == 0:
            self.logger.warning(f"No relevant documents found for query: {query} in project_id: {project.project_id}")
            return answer,full_prompt, chat_history
        
        # step 2: construct prompt with retrieved documents and the query

        system_prompt = self.template_parser.get("rag", "system_prompt")

        

        document_prompt="\n".join([
            self.template_parser.get("rag", "document_prompt",{
                "doc_num": i+1,
                "chunk_text": self.generation_client.process_text(doc.text),
            })
            for i, doc in enumerate(retrieved_document)
        ])

        footer_prompt = "\n".join([
            self.template_parser.get("rag", "footer_prompt", {
                "query": query
            })
        ])

        chat_history=[
            self.generation_client.construct_prompt(
               prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value),
        ]

        full_prompt= "\n\n".join([document_prompt, footer_prompt])

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer,full_prompt, chat_history

        
        
