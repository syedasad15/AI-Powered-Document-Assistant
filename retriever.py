from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # updated import if needed
import traceback
from typing import List
import faiss
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import traceback
from typing import List

from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import traceback
from typing import List

# def create_vector_store(chunks: List) -> FAISS:
#     try:
#         print(f"🔢 Number of chunks to embed: {len(chunks)}")
#         if hasattr(chunks[0], 'page_content'):
#             texts = [doc.page_content for doc in chunks]
#         else:
#             texts = chunks
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         print("⚙️ Creating vector store using FAISS...")
#         vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)
#         print("✅ Vector store created successfully.")
#         return vector_store
#     except Exception as e:
#         print("❌ Error while creating vector store:", e)
#         traceback.print_exc()
#         return None
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import traceback
from typing import List

def create_vector_store(chunks: List) -> FAISS:
    try:
        print(f"🔢 Number of chunks to embed: {len(chunks)}")
        if hasattr(chunks[0], 'page_content'):
            texts = [doc.page_content for doc in chunks]
        else:
            texts = chunks
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        print("⚙️ Creating vector store using FAISS...")
        vector_store = FAISS.from_texts(texts=texts, embedding=embeddings)
        print("✅ Vector store created successfully.")
        return vector_store
    except Exception as e:
        print("❌ Error while creating vector store:", e)
        traceback.print_exc()
        return None