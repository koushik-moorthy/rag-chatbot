#!/usr/bin/env python3

import os
import sys
import boto3
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configuration
DATA_DIR = os.getenv("RAG_DATA_DIR", "./data")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
COSINE_THRESHOLD = 0.6
MAX_TOKENS = 512
TEMPERATURE = 0.0
MAX_HISTORY = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# STEP 1: DOCUMENT LOADING FUNCTIONS
# ============================================================================

# Load plain text file
def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Extract text from PDF file
def load_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# Extract text from DOCX file
def load_docx(file_path):
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

# Load document based on file extension
def load_document(file_path):
    extension = file_path.lower().split('.')[-1]
    if extension == "txt":
        return load_txt(file_path)
    elif extension == "pdf":
        return load_pdf(file_path)
    elif extension == "docx":
        return load_docx(file_path)
    else:
        return ""

# ============================================================================
# STEP 2: TEXT CHUNKING
# ============================================================================

# Split text into overlapping chunks for better retrieval
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text.strip():
        return []
    
    tokens = text.split()
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(tokens):
            break
    
    return chunks

# ============================================================================
# STEP 3: DOCUMENT INDEXING
# ============================================================================

# Load all supported documents from directory and chunk them
def load_and_chunk_documents(data_directory):
    print(f"Loading documents from: {data_directory}")
    
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"Data directory '{data_directory}' not found")
    
    all_chunks = []
    supported_extensions = {'.txt', '.pdf', '.docx'}
    
    for filename in os.listdir(data_directory):
        if not any(filename.lower().endswith(ext) for ext in supported_extensions):
            continue
        
        file_path = os.path.join(data_directory, filename)
        print(f"Processing: {filename}")
        
        content = load_document(file_path)
        if content:
            chunks = chunk_text(content)
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
    
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks

# ============================================================================
# STEP 4: VECTOR DATABASE SETUP
# ============================================================================

# Initialize ChromaDB collection for vector storage
def initialize_vector_store(collection_name="documents"):
    print("Initializing vector database...")
    client = chromadb.Client(Settings())
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Vector database ready")
    return collection, embedder

# Convert text chunks to embeddings and store in vector database
def index_documents(collection, embedder, chunks):
    if not chunks:
        print("No chunks to index")
        return
    
    print("Generating embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    print("Storing in vector database...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )
    print(f"Indexed {len(chunks)} chunks successfully")

# Search for relevant documents using semantic similarity
def search_documents(collection, embedder, query, k=TOP_K_RESULTS):
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"]
    )
    return results["documents"][0], results["distances"][0]

# ============================================================================
# STEP 5: LLM INTEGRATION
# ============================================================================

# Generate response using Claude via AWS Bedrock
def generate_response(prompt):
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    try:
        response = bedrock_client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": TEMPERATURE, "maxTokens": MAX_TOKENS}
        )
        return response["output"]["message"]["content"][0]["text"]
    except ClientError as e:
        print(f"Error calling Bedrock: {e}")
        return "Sorry, I encountered an error while generating a response."

# ============================================================================
# STEP 6: PROMPT CONSTRUCTION
# ============================================================================

# Build prompt with conversation history and retrieved context
def build_prompt(history, query, documents, distances):
    # Include recent conversation history
    recent_history = history[-MAX_HISTORY:]
    history_text = ""
    
    if recent_history:
        history_text = "Conversation History:\n"
        for item in recent_history:
            history_text += f"User: {item['q']}\nBot: {item['a']}\n"
        history_text += "\n"
    
    # Check if retrieved documents are relevant enough
    # if distance is less than COSINE_THRESHOLD, we consider it relevant, if not, we use LLM knowledge only
    if not distances or distances[0] > COSINE_THRESHOLD:
        # Fall back to LLM knowledge only
        prompt = (
            "You are a helpful assistant. Use the conversation history and "
            "your own knowledge to answer the question.\n\n"
            f"{history_text}"
            f"Question: {query}\n"
            "Answer:"
        )
    else:
        # Use retrieved documents (RAG approach)
        context = "\n\n---\n\n".join(documents)
        prompt = (
            "You are a helpful assistant. Use the conversation history and "
            "the provided context to answer the question.\n\n"
            f"{history_text}"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
    
    return prompt

# ============================================================================
# STEP 7: QUERY PROCESSING
# ============================================================================

# Process a user query through the complete RAG pipeline
def process_query(collection, embedder, history, user_question):
    print(f"Processing query: {user_question}")
    
    # Retrieve relevant documents
    documents, distances = search_documents(collection, embedder, user_question)
    # print(f"Similarity scores: {[f'{d:.3f}' for d in distances]}")
    
    # Build prompt with context
    prompt = build_prompt(history, user_question, documents, distances)
    
    # Generate response
    response = generate_response(prompt)
    
    # Update conversation history
    history.append({"q": user_question, "a": response})
    
    return response

# ============================================================================
# STEP 8: CHAT INTERFACE
# ============================================================================

# Start interactive chat interface
def start_chat(collection, embedder):
    print("\n" + "="*50)
    print("RAG CHAT INTERFACE")
    print("="*50)
    print("Type 'exit' to quit")
    print()
    
    history = []
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = process_query(collection, embedder, history, user_input)
            print(f"Bot: {response}\n")
            
    except KeyboardInterrupt:
        print("\nChat interrupted. Goodbye!")

# ============================================================================
# STEP 9: MAIN SYSTEM SETUP AND EXECUTION
# ============================================================================

def main():
    try:
        print("🚀 Starting RAG System Setup...")
        
        # Step 1: Load and chunk documents
        chunks = load_and_chunk_documents(DATA_DIR)
        
        # Step 2: Initialize vector database
        collection, embedder = initialize_vector_store()
        
        # Step 3: Index documents
        if chunks:
            index_documents(collection, embedder, chunks)
        else:
            print("No documents found. System will use LLM knowledge only.")
        
        print("✅ RAG System ready!")
        
        # Step 4: Start chat interface
        start_chat(collection, embedder)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure your data directory exists and contains supported files.")
        sys.exit(1)
    except Exception as e:
        print(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()