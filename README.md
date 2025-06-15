# RAG System - Chat with Your Documents

A simple Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using AWS Bedrock Claude.

## Features

- Supports TXT, PDF, DOCX files
- Semantic search using vector embeddings
- Conversational chat with memory
- AWS Bedrock Claude integration
- Smart fallback to LLM knowledge when documents aren't relevant

## Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Setup AWS Credentials
```bash
aws configure
# or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### 3. Prepare Your Documents
```bash
mkdir data
# Add your .txt, .pdf, or .docx files to the data/ folder
```

### 4. Run the System
```bash
python3 rag_chat.py
```

## Configuration

Set environment variables (optional):

```bash
export RAG_DATA_DIR="./data"                           # Document directory
export BEDROCK_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"  # Claude model
export AWS_REGION="us-east-1"                          # AWS region
```

## How It Works

1. **Document Processing**: Loads and chunks your documents
2. **Vector Indexing**: Creates embeddings using sentence-transformers
3. **Storage**: Stores vectors in ChromaDB for fast retrieval
4. **Chat**: Searches relevant chunks and generates responses with Claude

## File Structure

```
project/
├── rag_chat.py      # Main RAG system
├── requirements.txt   # Python dependencies
├── README.md         # This file
└── data/             # Your documents folder
    ├── document1.pdf
    ├── document2.txt
    └── document3.docx
```

## Example Usage

```
You: What is the main topic of the documents?
Bot: Based on the documents, the main topics cover...

You: Can you summarize the key points?
Bot: Here are the key points from your documents:
1. ...
2. ...

You: exit
```

## Troubleshooting

**No documents found**: Make sure your `data/` folder contains .txt, .pdf, or .docx files

**AWS errors**: Check your AWS credentials and region settings

**Memory issues**: Reduce `CHUNK_SIZE` in the configuration section
