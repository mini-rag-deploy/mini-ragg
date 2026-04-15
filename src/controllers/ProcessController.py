from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnum
from typing import List
from dataclasses import dataclass
from ingestion.loaders import DocumentLoader, RawDocument
from ingestion.ocr import OCREngine
import logging

logger = logging.getLogger('uvicorn.error')

@dataclass
class Document:
    page_content: str
    metadata: dict

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.document_loader = DocumentLoader()
        # Initialize OCR engine for Arabic + English
        self.ocr_engine = OCREngine(lang="ara+eng", dpi=300, min_confidence=40)


    def get_file_extension(self,file_id:str):
        return os.path.splitext(file_id)[-1]
    

    def get_file_loader(self, file_id:str):
        """
        DEPRECATED: Use get_file_content() which now uses the new DocumentLoader
        """
        file_path = os.path.join(self.project_path, file_id)
        
        if not os.path.exists(file_path):
            return None
        
        # Use new DocumentLoader for all file types
        return self.document_loader
    
    def get_file_content(self, file_id:str):
        """
        Load file content using the new multi-format DocumentLoader.
        Supports: PDF, DOCX, PPTX, TXT, Images (with OCR)
        
        Now includes OCR processing for:
        - Scanned PDF pages (image-only pages)
        - Embedded images in documents
        - Standalone image files
        """
        file_path = os.path.join(self.project_path, file_id)
        
        if not os.path.exists(file_path):
            logger.error(f"[ProcessController] File not found: {file_path}")
            return None
        
        try:
            # Step 1: Load documents using DocumentLoader
            raw_docs: List[RawDocument] = self.document_loader.load_file(file_path)
            
            if not raw_docs:
                logger.warning(f"[ProcessController] No content extracted from: {file_id}")
                return None
            
            # Step 2: Apply OCR to documents that need it
            docs_needing_ocr = [doc for doc in raw_docs if doc.needs_ocr]
            if docs_needing_ocr:
                logger.info(f"[ProcessController] {len(docs_needing_ocr)} pages need OCR in {file_id}")
                try:
                    # Process OCR (pass source path for PDF pages)
                    raw_docs = self.ocr_engine.process_documents(raw_docs, source_path=file_path)
                    logger.info(f"[ProcessController] OCR completed for {file_id}")
                except Exception as ocr_exc:
                    logger.error(f"[ProcessController] OCR failed for {file_id}: {ocr_exc}")
                    # Continue with non-OCR documents
            
            # Step 3: Convert RawDocument to LangChain Document format
            documents = []
            for raw_doc in raw_docs:
                # Skip empty documents
                if raw_doc.is_empty():
                    logger.debug(f"[ProcessController] Skipping empty document from page {raw_doc.metadata.get('page')}")
                    continue
                
                documents.append(Document(
                    page_content=raw_doc.text,
                    metadata=raw_doc.metadata
                ))
            
            logger.info(f"[ProcessController] Loaded {len(documents)} documents from {file_id} "
                       f"({len([d for d in raw_docs if d.metadata.get('ocr_applied')])} with OCR)")
            return documents
            
        except Exception as exc:
            logger.error(f"[ProcessController] Error loading {file_id}: {exc}")
            return None
    
    def process_file_content(self , file_id:str, file_content:list, chunk_size:int, overlap_size:int):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size,length_function=len)

        file_content_texts = [doc.page_content for doc in file_content]

        file_content_metadata = [doc.metadata for doc in file_content]

        chunks = text_splitter.create_documents(file_content_texts,
                                                metadatas=file_content_metadata
                                                )
        # chunks = self.process_simpler_splitter(texts=file_content_texts, metadata=file_content_metadata, chunk_size=chunk_size)

        return chunks
    
    def process_simpler_splitter(self,texts: List[str], metadata: List, chunk_size:int, splitter_tag: str="\n"):

        full_text = " ".join(texts)

        # split by \n
        lines = [doc.strip() for doc in full_text.split(splitter_tag) if len(doc.strip())>1]

        chunks =[]
        current_chunk = ""

        for line in lines:
            current_chunk += line + splitter_tag
            if len(current_chunk) >= chunk_size:

                chunks.append(Document(page_content=current_chunk.strip(), metadata={}))
                
                current_chunk = ""
        
        if len(current_chunk)>0:
            chunks.append(Document(page_content=current_chunk.strip(), metadata={}))

        
        return chunks





