from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnum
from typing import List
from dataclasses import dataclass
from ingestion.loaders import DocumentLoader, RawDocument
from ingestion.ocr import OCREngine
from ingestion.hybrid_chunker import HybridChunker, create_hybrid_chunks
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
        # Initialize hybrid chunker for context-aware chunking
        self.hybrid_chunker = HybridChunker(
            chunk_size=512,
            chunk_overlap=64,
            min_chunk_size=60,
            deduplicate=True
        )


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
    
    def get_file_content(self, file_id: str):
        """
        Load file content using the new multi-format DocumentLoader.
        Supports: PDF, DOCX, PPTX, TXT, Images (with OCR)
        
        Now includes OCR processing for:
        - Scanned PDF pages (image-only pages)
        - Embedded images in documents
        - Standalone image files
        
        Returns RawDocument objects for hybrid chunking.
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
                logger.info(f"[ProcessController] {len(docs_needing_ocr)} documents need OCR in {file_id}")
                try:
                    # Process OCR (pass source path for PDF pages)
                    raw_docs = self.ocr_engine.process_documents(raw_docs, source_path=file_path)
                    logger.info(f"[ProcessController] OCR completed for {file_id}")
                except Exception as ocr_exc:
                    logger.error(f"[ProcessController] OCR failed for {file_id}: {ocr_exc}")
                    # Continue with non-OCR documents
            
            # Filter out empty documents
            valid_docs = [doc for doc in raw_docs if not doc.is_empty()]
            
            logger.info(f"[ProcessController] Loaded {len(valid_docs)} documents from {file_id} "
                       f"({len([d for d in raw_docs if d.metadata.get('ocr_applied')])} with OCR)")
            return valid_docs
            
        except Exception as exc:
            logger.error(f"[ProcessController] Error loading {file_id}: {exc}")
            return None
    
    def process_file_content(self, file_id: str, raw_documents: List[RawDocument], chunk_size: int, overlap_size: int):
        """
        Process file content using hybrid chunking to maintain text-image context.
        
        Args:
            file_id: File identifier
            raw_documents: List of RawDocument objects (mix of text and images)
            chunk_size: Target chunk size in characters
            overlap_size: Overlap between chunks
            
        Returns:
            List of Document objects with hybrid chunks
        """
        try:
            # Use hybrid chunker to create context-aware chunks
            self.hybrid_chunker.chunk_size = chunk_size
            self.hybrid_chunker.chunk_overlap = overlap_size
            
            # Create hybrid chunks that merge text and images contextually
            hybrid_chunks = self.hybrid_chunker.create_hybrid_documents(raw_documents)
            document_chunks = self.hybrid_chunker.chunk_hybrid_documents(hybrid_chunks)
            
            # Convert to LangChain Document format
            documents = []
            for chunk in document_chunks:
                documents.append(Document(
                    page_content=chunk.chunk_text,
                    metadata=chunk.metadata
                ))
            
            logger.info(f"[ProcessController] Created {len(documents)} hybrid chunks from {file_id}")
            return documents
            
        except Exception as exc:
            logger.error(f"[ProcessController] Error processing file content for {file_id}: {exc}")
            # Fallback to legacy chunking
            return self._fallback_chunking(raw_documents, chunk_size, overlap_size)
    
    def _fallback_chunking(self, raw_documents: List[RawDocument], chunk_size: int, overlap_size: int):
        """
        Fallback to legacy chunking if hybrid chunking fails.
        """
        logger.warning("[ProcessController] Using fallback chunking")
        
        # Convert to legacy format
        legacy_docs = []
        for raw_doc in raw_documents:
            if not raw_doc.is_empty():
                legacy_docs.append(Document(
                    page_content=raw_doc.text,
                    metadata=raw_doc.metadata
                ))
        
        # Use legacy chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap_size,
            length_function=len
        )
        
        file_content_texts = [doc.page_content for doc in legacy_docs]
        file_content_metadata = [doc.metadata for doc in legacy_docs]
        
        chunks = text_splitter.create_documents(
            file_content_texts,
            metadatas=file_content_metadata
        )
        
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





