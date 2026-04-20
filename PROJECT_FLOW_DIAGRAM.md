# Complete Project Flow with Hybrid Chunking

## 🏗️ **System Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MINI-RAG SYSTEM ARCHITECTURE                         │
│                          (With Hybrid Chunking)                                │
└─────────────────────────────────────────────────────────────────────────────────┘

📁 INPUT DOCUMENTS                    🔄 PROCESSING PIPELINE                    📊 RAG SYSTEM
┌─────────────────┐                  ┌─────────────────────┐                  ┌─────────────────┐
│ • PDF Files     │                  │                     │                  │                 │
│ • DOCX Files    │ ────────────────▶│   HYBRID CHUNKING   │ ────────────────▶│   RETRIEVAL     │
│ • PPTX Files    │                  │     PIPELINE        │                  │   & ANSWERING   │
│ • Images        │                  │                     │                  │                 │
│ • URLs          │                  └─────────────────────┘                  └─────────────────┘
└─────────────────┘                           │                                         │
                                              ▼                                         ▼
                                    ┌─────────────────────┐                  ┌─────────────────┐
                                    │   VECTOR DATABASE   │                  │   USER QUERIES  │
                                    │   (Qdrant + BM25)   │                  │   & RESPONSES   │
                                    └─────────────────────┘                  └─────────────────┘
```

## 📋 **Detailed Processing Flow**

### **Phase 1: Document Ingestion & Processing**

```
🔄 DOCUMENT PROCESSING PIPELINE
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  📄 INPUT FILE                                                                              │
│  (PDF/DOCX/PPTX/Image)                                                                     │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   DocumentLoader    │ ◄─── src/ingestion/loaders.py                                    │
│  │                     │                                                                   │
│  │ • PDFLoader         │ ──── Extracts text + embedded images                             │
│  │ • WordLoader        │ ──── Extracts paragraphs + images                                │
│  │ • PowerPointLoader  │ ──── Extracts slides + images                                    │
│  │ • ImageLoader       │ ──── Flags for OCR processing                                    │
│  │ • URLLoader         │ ──── Fetches web content                                         │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   RawDocument[]     │ ◄─── List of extracted documents                                 │
│  │                     │                                                                   │
│  │ Text Documents:     │ ──── Page text, metadata                                         │
│  │ • source_type: pdf  │                                                                   │
│  │ • text: "content"   │                                                                   │
│  │ • needs_ocr: false  │                                                                   │
│  │                     │                                                                   │
│  │ Image Documents:    │ ──── Embedded images                                              │
│  │ • source_type:      │                                                                   │
│  │   pdf_embedded_img  │                                                                   │
│  │ • text: ""          │                                                                   │
│  │ • needs_ocr: true   │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │    OCR Engine       │ ◄─── src/ingestion/ocr.py                                        │
│  │                     │                                                                   │
│  │ • Tesseract OCR     │ ──── Arabic + English support                                    │
│  │ • Image preprocessing│ ──── Denoise, threshold, deskew                                 │
│  │ • Confidence filter │ ──── Min 40% confidence                                          │
│  │ • Multi-format      │ ──── PDF pages, embedded images, standalone images              │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │ OCR-Processed Docs  │ ◄─── All documents now have text                                 │
│  │                     │                                                                   │
│  │ Text Documents:     │ ──── Original text preserved                                     │
│  │ • text: "content"   │                                                                   │
│  │                     │                                                                   │
│  │ Image Documents:    │ ──── Now have OCR text                                           │
│  │ • text: "OCR text"  │                                                                   │
│  │ • ocr_applied: true │                                                                   │
│  └─────────────────────┘                                                                   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### **Phase 2: Hybrid Chunking (NEW!)**

```
🥪 HYBRID CHUNKING PIPELINE
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  📋 OCR-Processed Documents                                                                 │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   HybridChunker     │ ◄─── src/ingestion/hybrid_chunker.py                             │
│  │                     │                                                                   │
│  │ Step 1: Group Docs  │ ──── Group by source + page                                      │
│  │ • Same PDF page     │                                                                   │
│  │ • Text + Images     │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │  HybridDocument[]   │ ◄─── Grouped documents                                           │
│  │                     │                                                                   │
│  │ Page 1:             │                                                                   │
│  │ • text: "content"   │ ──── Combined page text                                          │
│  │ • images: [         │                                                                   │
│  │   {text: "OCR1"},   │ ──── Associated images with OCR                                  │
│  │   {text: "OCR2"}    │                                                                   │
│  │ ]                   │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │ Step 2: Merge       │ ◄─── _merge_images_into_text()                                   │
│  │ Images into Text    │                                                                   │
│  │                     │                                                                   │
│  │ Strategy:           │                                                                   │
│  │ • Split text into   │ ──── Lines, sentences, or paragraphs                            │
│  │   logical sections  │                                                                   │
│  │ • Calculate optimal │ ──── Distribute images evenly                                    │
│  │   insertion points  │                                                                   │
│  │ • Insert at natural │ ──── After sentences/paragraphs                                  │
│  │   break points      │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   Enhanced Text     │ ◄─── Text with images inserted                                   │
│  │                     │                                                                   │
│  │ "The Wonders of     │                                                                   │
│  │  Astronomy...       │ ──── Original text                                               │
│  │                     │                                                                   │
│  │  [Image 1: Milky    │ ──── Image inserted in middle                                    │
│  │   Way contains      │                                                                   │
│  │   100 stars]        │                                                                   │
│  │                     │                                                                   │
│  │  Modern astronomy   │ ──── Continuation of text                                        │
│  │  is divided..."     │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │ Step 3: Chunk       │ ◄─── Context-aware chunking                                      │
│  │ Enhanced Text       │                                                                   │
│  │                     │                                                                   │
│  │ • chunk_size: 512   │ ──── Configurable size                                           │
│  │ • chunk_overlap: 64 │ ──── Overlap for context                                         │
│  │ • Sentence-aware    │ ──── Never split mid-sentence                                    │
│  │ • Deduplication     │ ──── Remove duplicate chunks                                     │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │  DocumentChunk[]    │ ◄─── Final hybrid chunks                                         │
│  │                     │                                                                   │
│  │ Chunk 1:            │                                                                   │
│  │ • chunk_text:       │ ──── Text + Image content                                        │
│  │   "Astronomy...     │                                                                   │
│  │    [Image: data]    │                                                                   │
│  │    Modern..."       │                                                                   │
│  │ • metadata:         │                                                                   │
│  │   - has_images: true│ ──── Rich metadata                                               │
│  │   - image_count: 1  │                                                                   │
│  │   - referenced_imgs │                                                                   │
│  └─────────────────────┘                                                                   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### **Phase 3: Indexing & Storage**

```
💾 INDEXING PIPELINE
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  📋 Hybrid Chunks                                                                          │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │  ProcessController  │ ◄─── src/controllers/ProcessController.py                        │
│  │                     │                                                                   │
│  │ • get_file_content()│ ──── Returns RawDocument[]                                       │
│  │ • process_file_     │ ──── Uses HybridChunker                                          │
│  │   content()         │                                                                   │
│  │ • Fallback to       │ ──── Legacy chunking if hybrid fails                             │
│  │   legacy chunking   │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   NLPController     │ ◄─── src/controllers/NLPController.py                            │
│  │                     │                                                                   │
│  │ • Embedding         │ ──── Convert chunks to vectors                                    │
│  │   Generation        │                                                                   │
│  │ • Vector DB         │ ──── Store in Qdrant                                             │
│  │   Indexing          │                                                                   │
│  │ • BM25 Index        │ ──── Keyword search index                                        │
│  │   Building          │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   Storage Layer     │                                                                   │
│  │                     │                                                                   │
│  │ Vector Database:    │                                                                   │
│  │ • Qdrant            │ ──── Semantic vectors                                            │
│  │ • Collections       │ ──── Per-project collections                                     │
│  │ • Metadata          │ ──── Rich chunk metadata                                         │
│  │                     │                                                                   │
│  │ Keyword Index:      │                                                                   │
│  │ • BM25              │ ──── Traditional keyword search                                  │
│  │ • Term frequencies  │ ──── Statistical relevance                                       │
│  │                     │                                                                   │
│  │ Database:           │                                                                   │
│  │ • PostgreSQL        │ ──── Project/asset metadata                                      │
│  │ • Alembic           │ ──── Schema migrations                                           │
│  └─────────────────────┘                                                                   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### **Phase 4: Query Processing & Retrieval**

```
🔍 RETRIEVAL PIPELINE
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  💬 User Query                                                                              │
│  "How many stars are in the Milky Way?"                                                    │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   Query Processing  │ ◄─── src/routes/nlp.py                                           │
│  │                     │                                                                   │
│  │ • Multi-query       │ ──── Generate query variations                                    │
│  │   generation        │                                                                   │
│  │ • Query embedding   │ ──── Convert to vector                                           │
│  │ • Preprocessing     │ ──── Clean and normalize                                         │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │  Hybrid Search      │ ◄─── src/retrieval/hybrid_search.py                              │
│  │                     │                                                                   │
│  │ Semantic Search:    │                                                                   │
│  │ • Vector similarity │ ──── Cosine similarity in Qdrant                                 │
│  │ • Top-k results     │                                                                   │
│  │                     │                                                                   │
│  │ Keyword Search:     │                                                                   │
│  │ • BM25 scoring      │ ──── Traditional keyword matching                                │
│  │ • Term frequency    │                                                                   │
│  │                     │                                                                   │
│  │ Fusion:             │                                                                   │
│  │ • RRF (Reciprocal   │ ──── Combine semantic + keyword                                  │
│  │   Rank Fusion)      │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   Reranking         │ ◄─── src/retrieval/reranker.py                                   │
│  │                     │                                                                   │
│  │ • Cohere Rerank     │ ──── Advanced relevance scoring                                  │
│  │ • Cross-encoder     │ ──── Query-document interaction                                  │
│  │ • Final ranking     │ ──── Best results first                                          │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │  Retrieved Chunks   │ ◄─── Hybrid chunks with context                                  │
│  │                     │                                                                   │
│  │ Top Result:         │                                                                   │
│  │ "The Wonders of     │                                                                   │
│  │  Astronomy...       │ ──── Text context                                                │
│  │                     │                                                                   │
│  │  [Image 1: Milky    │ ──── Image with OCR data                                         │
│  │   Way contains      │                                                                   │
│  │   100 stars]        │                                                                   │
│  │                     │                                                                   │
│  │  Modern astronomy   │ ──── More context                                                │
│  │  is divided..."     │                                                                   │
│  │                     │                                                                   │
│  │ Score: 0.9990601    │ ──── High relevance                                              │
│  └─────────────────────┘                                                                   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### **Phase 5: Answer Generation**

```
🤖 ANSWER GENERATION PIPELINE
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  📋 Retrieved Chunks + 💬 User Query                                                       │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   RAG Graph         │ ◄─── src/graph/rag_graph.py                                      │
│  │                     │                                                                   │
│  │ • LangGraph         │ ──── Workflow orchestration                                      │
│  │ • State management  │ ──── Track conversation state                                    │
│  │ • Multi-step        │ ──── Complex reasoning flows                                     │
│  │   reasoning         │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   LLM Processing    │ ◄─── Language Model                                              │
│  │                     │                                                                   │
│  │ Context:            │                                                                   │
│  │ "Based on this      │                                                                   │
│  │  document about     │ ──── Rich context with text + images                            │
│  │  astronomy that     │                                                                   │
│  │  mentions the Milky │                                                                   │
│  │  Way contains 100   │                                                                   │
│  │  stars..."          │                                                                   │
│  │                     │                                                                   │
│  │ Question:           │                                                                   │
│  │ "How many stars     │ ──── Original user query                                         │
│  │  are in the Milky   │                                                                   │
│  │  Way?"              │                                                                   │
│  └─────────────────────┘                                                                   │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────┐                                                                   │
│  │   Generated Answer  │ ◄─── Final response                                              │
│  │                     │                                                                   │
│  │ "According to the   │                                                                   │
│  │  document, the      │ ──── Accurate answer with source                                 │
│  │  Milky Way contains │                                                                   │
│  │  over 100 billion   │                                                                   │
│  │  stars, as shown    │                                                                   │
│  │  in the OCR data    │                                                                   │
│  │  from the image."   │                                                                   │
│  │                     │                                                                   │
│  │ Source: Page 1,     │ ──── Source attribution                                          │
│  │ Image 1             │                                                                   │
│  └─────────────────────┘                                                                   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 **Complete System Flow**

```
📄 INPUT → 🔄 PROCESSING → 💾 STORAGE → 🔍 RETRIEVAL → 🤖 GENERATION → 💬 OUTPUT

┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Upload  │──▶│ Hybrid  │──▶│ Vector  │──▶│ Hybrid  │──▶│   LLM   │──▶│ Answer  │
│ Files   │   │Chunking │   │Database │   │ Search  │   │Process  │   │ User    │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
     │             │             │             │             │             │
     ▼             ▼             ▼             ▼             ▼             ▼
• PDF/DOCX    • Text+Image   • Qdrant      • Semantic    • Context     • Rich
• PPTX/IMG    • Contextual   • BM25        • Keyword     • Reasoning   • Accurate
• URLs        • Chunks       • PostgreSQL  • Reranking   • Sources     • Sourced
```

## 🎯 **Key Improvements with Hybrid Chunking**

### **Before (Old System):**
```
❌ Text Chunk: "The Wonders of Astronomy..."
❌ Image Chunk: "Milky Way contains 100 stars" (SEPARATE!)
❌ Context Lost: LLM doesn't see connection
❌ Poor Answers: Incomplete information
```

### **After (Hybrid System):**
```
✅ Hybrid Chunk: "The Wonders of Astronomy... [Image: Milky Way contains 100 stars] Modern astronomy..."
✅ Context Preserved: Text and image together
✅ Rich Retrieval: Complete information in one chunk
✅ Better Answers: LLM sees full context
```

## 📊 **System Benefits**

1. **🔗 Context Preservation**: Images stay with related text
2. **🎯 Better Retrieval**: One chunk = complete information  
3. **🤖 Improved LLM**: Sees text + image data together
4. **📈 Higher Accuracy**: More precise answers
5. **🔍 Rich Metadata**: Tracks image references per chunk
6. **⚡ Efficient**: No separate text/image processing
7. **🛡️ Robust**: Fallback to legacy chunking if needed

This hybrid chunking system transforms your RAG pipeline from fragmented text/image processing into a unified, context-aware system that provides much richer and more accurate responses! 🚀