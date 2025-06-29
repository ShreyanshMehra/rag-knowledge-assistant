# Retrieval-Augmented Generation (RAG) TechDocs Assistant  
### Using LangChain, ChromaDB, and MiniLM

## Introduction

This Retrieval-Augmented Generation (RAG) application lets you build powerful, interactive AI chatbots and Q&A tools for your own documents, books, or files. It is designed for **cost-effective, local deployment** using the **MiniLM (all-MiniLM-L6-v2) model** for embeddings, LangChain for orchestration, and ChromaDB for vector storage.

**Example use cases:**
1. Ask questions about a large collection of technical documentation or notes.
2. Build a customer support chatbot that answers based on your specific instructions or manuals.

## Project Workflow

1. **Document Ingestion & Chunking:**  
   Your documents (e.g., in Markdown format) are split into manageable text chunks (typically 512–1024 characters) for efficient retrieval and context-aware answering.

2. **Embedding & Vector Storage:**  
   Each chunk is embedded using the free, open-source [MiniLM (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model via Hugging Face, and stored in a local [ChromaDB](https://www.trychroma.com/) vector database.

3. **Query & Retrieval:**  
   When you ask a question, the system retrieves the most relevant document chunks using semantic similarity search.

4. **Answer Generation:**  
   A local LLM (e.g., FLAN-T5 via Hugging Face) generates an answer based on the retrieved context, and returns both the answer and the sources.

## Example Output

![Example Output](https://github.com/mlsmall/RAG-Application-with-LangChain/blob/main/output.png)

## Instructions

### 1. Fork the Repository
### 2. Download the Repository
```
git clone https://github.com/yourusername/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
*Make sure you have a CUDA-enabled PyTorch if you want GPU acceleration for embeddings.*

### 4. Add Your Documents

- Place your `.md` (Markdown) or text files inside the `data` directory.
- You can use any tech documentation, cheat sheets, or notes you want to query.

### 5. Initialize the RAG Database
```
python initialize_rag_db.py
```
This script will split your documents, create MiniLM embeddings, and store them in ChromaDB.

### 6. Ask Questions

```
python ask_rag.py "Your question here"
```

## Features

- **Fast and scalable:** Uses MiniLM for embeddings and ChromaDB for vector storage.
- **Flexible:** Works with any technical documentation or notes in Markdown/text format.
- **Easy to extend:** Swap in new datasets or LLMs as needed.

## File Overview

- `initialize_rag_db.py` — Ingests and embeds your documents, builds the vector database.
- `ask_rag.py` — Queries your vector database and generates answers using a local LLM.

## Credits & References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [MiniLM Model (Hugging Face)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)



