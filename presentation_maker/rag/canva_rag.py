"""
Canva RAG - Retrieval Augmented Generation for Canva SDK Documentation

This module queries the Pinecone vector database to retrieve relevant
documentation for generating Canva design functions.

The embeddings are pre-computed and stored in Pinecone (see init_pinecone.py).
This module only queries the existing vectors - no embedding generation needed!
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import dotenv

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "canva-docs")

# Initialize embeddings model (used for query embedding only)
_embeddings = None
_vectorstore = None


def _get_vectorstore() -> PineconeVectorStore:
    """
    Get or create the Pinecone vector store instance.
    This is initialized once and reused for all queries.
    """
    global _embeddings, _vectorstore
    
    if _vectorstore is None:
        if not PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        # Initialize embeddings (same model and dimensions used during indexing)
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
        
        # Connect to existing Pinecone index
        _vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=_embeddings
        )
        
        print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    
    return _vectorstore


def query_pinecone(query: str, top_k: int = 1) -> list:
    """
    Query the Pinecone vector store for relevant documents.
    
    Args:
        query: The search query string
        top_k: Number of top results to return
        
    Returns:
        List of relevant Document objects
    """
    vectorstore = _get_vectorstore()
    
    # Perform similarity search
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    
    print("Retrieved documents:")
    for doc in retrieved_docs:
        print(f"  - Tab: {doc.metadata.get('tab', 'unknown')}")
    print("=" * 40)
    
    return retrieved_docs


def handle_rag(request_input: list) -> list:
    """
    Main RAG handler function.
    
    Takes a list of RAG queries and returns relevant documentation
    from the Pinecone vector store.
    
    Args:
        request_input: List of query strings (e.g., ["Images", "Text", "Tables"])
        
    Returns:
        List of relevant Document objects
    """
    # Determine how many unique results to fetch
    top_k = len(set(request_input))
    
    # Join queries into a single search string
    query = ",".join(request_input)
    
    print(f"RAG Query: {query}")
    print(f"Fetching top {top_k} results from Pinecone...")
    
    # Query Pinecone (no file reading, no embedding generation!)
    docs = query_pinecone(query, top_k)
    
    print(f"THE DOCUMENT VALUE IS: {docs}")
    
    return docs
