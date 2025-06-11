import re
import openai
from pprint import pprint


import os
from dotenv import load_dotenv

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()



def split_tabs(text):
    pattern = re.compile(r'<Tab name="([^"]+)">')
    stack = []
    chunks = []
    pos = 0

    for match in pattern.finditer(text):
        start = match.start()
        name = match.group(1)

        # If inside another tab, cut out content
        if stack:
            chunks.append((stack[-1], text[pos:start]))
        stack.append(name)
        pos = start

    if stack:
        chunks.append((stack[-1], text[pos:]))

    # print("+++++++++++++++++++++++++++++++++++++++")
    # print("FIRST 3 CHUNKS:")
    # print("+++++++++++++++++++++++++++++++++++++++")
    # for name, content in chunks[:2]:
    #     print(f"Tab: {name}")
    #     print("Content:")
    #     print(content)
    #     print("=======================================")
    # print("+++++++++++++++++++++++++++++++++++++++")

    return chunks


def load_chunks_to_vectorstore(chunks , query , top_k = 1):
    docs = [Document(page_content=content, metadata={"tab": name}) for name, content in chunks]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=24000, chunk_overlap=200     )
    docs = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    print("Retrieved documents:")
    for doc in retrieved_docs:
        # print(doc.page_content)
        print(doc.metadata)
        print("=======================================")
    return retrieved_docs


def handle_rag(request_input):
    top_k = len(set(request_input))
    query = ",".join(request_input)
    file_path = "./presentation_maker/rag/addElementAtPoint.txt"
    text = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    chunks = split_tabs(text)
    doc = load_chunks_to_vectorstore(chunks , query, top_k)
    print("THE DOCUMENT VALUE IS : " , doc)
    return doc

