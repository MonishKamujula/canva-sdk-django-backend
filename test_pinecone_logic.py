import sys
import os

# Add the current directory to sys.path to make imports work
sys.path.append(os.getcwd())

from presentation_maker.rag.init_pinecone import split_tabs, create_documents

def test_logic():
    text = """<Tab name="Tab1">
Content 1
</Tab>
<Tab name="Tab2">
Content 2
</Tab>
"""
    
    print("Testing split_tabs...")
    chunks = split_tabs(text)
    print(f"Chunks found: {len(chunks)}")
    for name, content in chunks:
        print(f" - {name}: {len(content)} chars")

    print("\nTesting create_documents...")
    docs = create_documents(chunks)
    print(f"Documents created: {len(docs)}")
    
    if len(docs) == len(chunks):
        print("\nSUCCESS: Number of documents matches number of chunks.")
    else:
        print(f"\nFAILURE: Created {len(docs)} documents from {len(chunks)} chunks.")

if __name__ == "__main__":
    test_logic()
