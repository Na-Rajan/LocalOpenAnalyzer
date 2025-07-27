🏠 Local RAG - Completely Offline Document Q&A
A fully local, private RAG (Retrieval-Augmented Generation) system that works entirely offline using local models.

🔒 100% Private & Local
No API keys required
No internet connection needed after setup
All processing happens on your machine
Your documents never leave your computer
🚀 Quick Start
1. Install Dependencies
bash
pip install -r requirements.txt
2. Run the Application
bash
streamlit run main.py
3. First Run Setup
Models will auto-download on first use (~500MB total)
Embedding model: ~90MB
LLM model: ~350MB
After download, works completely offline!
📋 Features
🏠 Local Models
Embedding: SentenceTransformers (all-MiniLM-L6-v2)
LLM: Local transformers models (DialoGPT, DistilGPT2, GPT2)
Vector DB: FAISS (no server required)
Processing: PyTorch (CPU/GPU support)
📄 Document Support
PDF files
Word documents (.docx)
Text files (.txt)
Smart text chunking with overlap
🎯 RAG Pipeline
Document upload and processing
Vector embeddings generation
Similarity search
Context-aware answer generation
Source attribution with confidence scores
🛠 Usage Guide
Step 1: Upload Documents
Click "Upload documents"
Select PDF, DOCX, or TXT files
Multiple files supported
Step 2: Build Index
Click "Build Index" to process documents
First run downloads models (one-time setup)
Creates searchable vector database
Step 3: Ask Questions
Enter questions in natural language
Get AI-generated answers with sources
View confidence scores and document references
⚙️ Configuration Options
Models
Embedding Models:
all-MiniLM-L6-v2 (default, fastest)
all-mpnet-base-v2 (better quality)
all-MiniLM-L12-v2 (balanced)
LLM Models:
DialoGPT-small (conversational)
DistilGPT2 (compact)
GPT2 (classic)
Processing
Chunk Size: 200-1000 characters
Max Results: 1-10 relevant documents
Answer Length: 50-300 tokens
💻 System Requirements
Minimum
Python 3.8+
4GB RAM
2GB disk space (for models)
CPU processing
Recommended
Python 3.9+
8GB RAM
NVIDIA GPU (CUDA support)
SSD storage
🚀 Performance Tips
For CPU Users
Use smaller models (DialoGPT-small)
Reduce chunk size to 300-400
Limit max results to 3-5
For GPU Users
Enable CUDA acceleration automatically
Use larger models for better quality
Process larger document collections
📂 File Structure
local-rag/
├── main.py              # Main application
├── requirements.txt     # Dependencies
├── README.md           # This guide
└── cache/              # Auto-created
    ├── uploaded_files/ # Temporary uploads
    └── models/         # Downloaded models
🔧 Troubleshooting
Model Download Issues
bash
# Manual model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
Memory Issues
Reduce chunk size in settings
Use smaller models
Process fewer documents at once
Close other applications
GPU Issues
bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
Common Fixes
Slow responses: Use CPU-optimized models
Poor answers: Increase context window
Memory errors: Reduce batch sizes
Model errors: Clear cache and re-download
🎯 Use Cases
Personal
Research paper analysis
Legal document review
Technical documentation Q&A
Book and article exploration
Professional
Internal knowledge base
Training material Q&A
Policy and procedure lookup
Code documentation search
Educational
Study guide creation
Literature analysis
Research assistance
Note organization
🔒 Privacy Features
No Network Calls: After initial setup
Local Storage: All data stays on device
No Logging: No usage data collected
Secure Processing: Documents processed locally
Cache Control: Manual cache clearing available
🚀 Advanced Usage
Custom Models
python
# In config, change models:
config = {
    "embedding_model": "your-custom-model",
    "llm_model": "your-llm-model"
}
Batch Processing
Upload multiple documents at once
Bulk index building
Batch question answering
Export Results
Copy answers and sources
Save Q&A sessions locally
Export processed chunks
🔄 Updates & Maintenance
Model Updates
Manually clear cache to re-download
Check for newer model versions
Backup important configurations
Storage Management
Clear cache periodically
Remove old uploaded files
Monitor disk usage
🎉 Ready to Go!
Your completely private, local RAG system is ready. No APIs, no subscriptions, no data sharing - just you and your documents!

Download → Install → Upload → Ask ✨

