"""
Pinecone Index Initialization Script

This script reads the Canva SDK documentation, generates embeddings,
and stores them in Pinecone for persistent vector storage.

Run this script ONCE to populate the Pinecone index.
After that, the RAG system will query the existing vectors.

Usage:
    python presentation_maker/rag/init_pinecone.py
"""

import os
import re
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.schema import Document
import dotenv

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "canva-docs")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Embedding dimensions for text-embedding-3-small (full native output)
EMBEDDING_DIMENSION = 1536


def split_tabs(text: str) -> list[tuple[str, str]]:
    """
    Split the documentation text by <Tab name="..."> sections.
    
    Returns a list of tuples: (tab_name, content)
    """
    pattern = re.compile(r'<Tab name="([^"]+)">')
    stack = []
    chunks = []
    pos = 0

    for match in pattern.finditer(text):
        start = match.start()
        name = match.group(1)

        if stack:
            chunks.append((stack[-1], text[pos:start]))
        stack.append(name)
        pos = start

    if stack:
        chunks.append((stack[-1], text[pos:]))

    return chunks


def create_documents(chunks: list[tuple[str, str]]) -> list[Document]:
    """
    Convert chunks into LangChain Document objects with metadata.
    """
    docs = [
        Document(page_content=content, metadata={"tab": name}) 
        for name, content in chunks
    ]
    
    return docs


def init_pinecone_index():
    """
    Initialize the Pinecone index. Creates it if it doesn't exist.
    """
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Free tier region
            )
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created successfully!")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")
    
    return pc.Index(PINECONE_INDEX_NAME)


def populate_index():
    """
    Main function to populate the Pinecone index with document embeddings.
    """
    print("=" * 60)
    print("Pinecone Index Initialization")
    print("=" * 60)
    
    # Read the documentation file
    file_path = os.path.join(os.path.dirname(__file__), "addElementAtPoint.txt")
    
    print(f"\n1. Reading documentation from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    print(f"   Loaded {len(text)} characters")
    
    # Split into tab-based chunks
    print("\n2. Splitting document by tabs...")
    chunks = split_tabs(text)
    print(f"   Found {len(chunks)} tab sections")
    for name, _ in chunks:
        print(f"      - {name}")
    
    # Create documents
    print("\n3. Creating document chunks...")
    docs = create_documents(chunks)
    print(f"   Created {len(docs)} document chunks")
    
    # Initialize Pinecone
    print("\n4. Initializing Pinecone index...")
    init_pinecone_index()
    
    # Create embeddings and store in Pinecone
    print("\n5. Generating embeddings and uploading to Pinecone...")
    print("   (This may take a moment...)")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=EMBEDDING_DIMENSION)
    
    # Use PineconeVectorStore to handle embedding and upserting
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    print("\n" + "=" * 60)
    print("SUCCESS! Pinecone index populated with document embeddings.")
    print("=" * 60)
    print(f"\nIndex Name: {PINECONE_INDEX_NAME}")
    print(f"Total Documents: {len(docs)}")
    print("\nYou can now use the RAG system without regenerating embeddings!")
    
    return vectorstore


if __name__ == "__main__":
    populate_index()
