#!/usr/bin/env python3
"""
Local RAG (Retrieval-Augmented Generation) Application
A completely local, self-contained RAG system using local models
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from PyPDF2 import PdfReader
import docx
import hashlib
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalRAG:
    """Completely local RAG system using local models"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "microsoft/DialoGPT-medium",  # Small conversational model
            "chunk_size": 500,
            "chunk_overlap": 50,
            "max_results": 5,
            "max_new_tokens": 150,
            "cache_dir": "./cache",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        if config:
            self.config.update(config)
            
        self.embedder = None
        self.llm_pipeline = None
        self.tokenizer = None
        self.index = None
        self.documents = []
        self.embeddings = None
        self._setup_cache()
        
    def _setup_cache(self):
        """Setup cache directory"""
        Path(self.config["cache_dir"]).mkdir(exist_ok=True)
        
    def _load_embedder(self):
        """Lazy load embedding model"""
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer(self.config["embedding_model"])
                logger.info(f"Loaded embedding model: {self.config['embedding_model']}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
                
    def _load_llm(self):
        """Lazy load local LLM"""
        if self.llm_pipeline is None:
            try:
                # Load a small local model suitable for QA
                model_name = self.config["llm_model"]
                
                # Try different model options based on availability
                model_options = [
                    "microsoft/DialoGPT-small",
                    "distilgpt2",
                    "gpt2"
                ]
                
                for model in [model_name] + model_options:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                            
                        self.llm_pipeline = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=self.tokenizer,
                            device=0 if self.config["device"] == "cuda" else -1,
                            torch_dtype=torch.float16 if self.config["device"] == "cuda" else torch.float32
                        )
                        logger.info(f"Loaded LLM model: {model}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model}: {e}")
                        continue
                        
                if self.llm_pipeline is None:
                    raise Exception("Failed to load any LLM model")
                    
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                raise
    
    def _chunk_text(self, text: str) -> List[str]:
        """Simple text chunking with overlap"""
        if not text.strip():
            return []
            
        words = text.split()
        chunks = []
        chunk_size = self.config["chunk_size"]
        overlap = self.config["chunk_overlap"]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
                
        return chunks
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from various file formats"""
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext == '.pdf':
                reader = PdfReader(file_path)
                return "\n".join([page.extract_text() for page in reader.pages])
            elif ext == '.docx':
                doc = docx.Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def add_document(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """Add a document to the knowledge base"""
        try:
            text = self._extract_text(file_path)
            if not text:
                return False
                
            chunks = self._chunk_text(text)
            if not chunks:
                return False
                
            doc_id = hashlib.md5(file_path.encode()).hexdigest()
            
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    "id": f"{doc_id}_{i}",
                    "text": chunk,
                    "source": file_path,
                    "chunk_index": i,
                    "metadata": metadata or {}
                })
                
            logger.info(f"Added {len(chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def build_index(self):
        """Build FAISS index from documents"""
        if not self.documents:
            raise ValueError("No documents to index")
            
        self._load_embedder()
        
        # Generate embeddings
        texts = [doc["text"] for doc in self.documents]
        self.embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Built index with {len(self.documents)} documents")
    
    def search(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Search for relevant documents"""
        if not self.index or not self.embedder:
            raise ValueError("Index not built. Call build_index() first.")
            
        k = k or self.config["max_results"]
        
        # Embed query
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result["score"] = float(score)
                results.append(result)
                
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using local LLM"""
        self._load_llm()
        
        # Prepare context
        context = "\n".join([doc["text"] for doc in context_docs[:3]])  # Limit context
        
        # Create prompt for QA
        prompt = f"""Context: {context[:800]}
        
Question: {query}
Answer:"""
        
        try:
            # Generate response
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=self.config["max_new_tokens"],
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_full_text=False
            )
            
            answer = response[0]['generated_text'].strip()
            
            # Clean up the answer
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
                
            return answer if answer else "I cannot find a relevant answer in the provided context."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer. Please try a simpler question."
    
    def query(self, question: str) -> Dict[str, Any]:
        """Complete RAG pipeline"""
        try:
            # Search for relevant documents
            relevant_docs = self.search(question)
            
            if not relevant_docs:
                return {
                    "question": question,
                    "answer": "No relevant documents found.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Generate answer
            answer = self.generate_answer(question, relevant_docs)
            
            # Calculate confidence (average similarity score)
            confidence = np.mean([doc["score"] for doc in relevant_docs])
            
            return {
                "question": question,
                "answer": answer,
                "sources": [{"source": doc["source"], "score": doc["score"]} for doc in relevant_docs],
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error in query pipeline: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {e}",
                "sources": [],
                "confidence": 0.0
            }

def main():
    """Streamlit UI"""
    st.set_page_config(page_title="Local RAG", page_icon="🏠", layout="wide")
    
    st.title("🏠 Local RAG - Offline Document Q&A")
    st.markdown("Upload documents and ask questions - completely local and private!")
    
    # Initialize session state
    if "rag" not in st.session_state:
        st.session_state.rag = LocalRAG()
        st.session_state.indexed = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Models")
        embedding_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "sentence-transformers/all-MiniLM-L12-v2"
        ]
        
        llm_models = [
            "microsoft/DialoGPT-small",
            "distilgpt2",
            "gpt2"
        ]
        
        selected_embedding = st.selectbox("Embedding Model", embedding_models)
        selected_llm = st.selectbox("LLM Model", llm_models)
        
        # Processing settings
        st.subheader("Processing")
        chunk_size = st.slider("Chunk Size", 200, 1000, 500)
        max_results = st.slider("Max Results", 1, 10, 5)
        max_tokens = st.slider("Max Answer Tokens", 50, 300, 150)
        
        # Device info
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ Using: {device}")
        
        # Update config
        st.session_state.rag.config.update({
            "embedding_model": selected_embedding,
            "llm_model": selected_llm,
            "chunk_size": chunk_size,
            "max_results": max_results,
            "max_new_tokens": max_tokens
        })
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents", 
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = f"./cache/{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                
                # Add to RAG system
                if st.session_state.rag.add_document(temp_path):
                    st.success(f"✅ Added: {file.name}")
                else:
                    st.error(f"❌ Failed to add: {file.name}")
        
        # Document stats
        if st.session_state.rag.documents:
            st.info(f"📊 Total chunks: {len(st.session_state.rag.documents)}")
        
        if st.button("🔨 Build Index", type="primary"):
            if st.session_state.rag.documents:
                with st.spinner("Building index... (First time may take longer to download models)"):
                    try:
                        st.session_state.rag.build_index()
                        st.session_state.indexed = True
                        st.success("✅ Index built successfully!")
                    except Exception as e:
                        st.error(f"❌ Error building index: {e}")
            else:
                st.warning("⚠️ No documents uploaded")
    
    with col2:
        st.header("❓ Ask Questions")
        
        if st.session_state.indexed:
            question = st.text_area("Enter your question:", height=100)
            
            if st.button("🔍 Search & Answer", type="primary") and question:
                with st.spinner("Searching documents and generating answer..."):
                    result = st.session_state.rag.query(question)
                    
                    st.subheader("🤖 Answer")
                    st.write(result["answer"])
                    
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result["sources"], 1):
                        st.write(f"{i}. {Path(source['source']).name} (Score: {source['score']:.3f})")
                    
                    st.subheader("📊 Confidence")
                    confidence_pct = result["confidence"] * 100
                    st.progress(result["confidence"])
                    st.write(f"Confidence: {confidence_pct:.1f}%")
            
            # Example questions
            st.subheader("💡 Example Questions")
            example_questions = [
                "What is the main topic of these documents?",
                "Can you summarize the key points?",
                "What are the important dates mentioned?",
                "Who are the main people or organizations discussed?"
            ]
            
            for eq in example_questions:
                if st.button(f"💬 {eq}", key=f"example_{eq}"):
                    st.session_state.example_question = eq
                    
        else:
            st.info("📋 Please upload documents and build the index first.")
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            1. **Upload Documents**: Add PDF, DOCX, or TXT files
            2. **Build Index**: Process and index your documents  
            3. **Ask Questions**: Query your documents in natural language
            4. **Get Answers**: Receive AI-generated answers with sources
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("🏠 **Completely Local & Private** - No data leaves your machine!")
    st.markdown("Built with ❤️ using Streamlit, Sentence Transformers, Transformers, and FAISS")

if __name__ == "__main__":
    main()