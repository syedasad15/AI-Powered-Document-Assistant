from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import re
import pdfplumber


def clean_text(text: str) -> str:
    """Clean extracted text by removing redundant spaces and special characters."""
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove multiple newlines
    text = re.sub(r'[^\w\s.,-]', '', text)  # Remove special characters
    return text.strip()

# def load_pdf(file_path: str) -> str:
#     raw_text = ""
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 raw_text += text + "\n"
#     return clean_text(raw_text)

# def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 300) -> List[Document]:
#     """Split text into structured, overlapping chunks for vector embedding."""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", ".", " ", r"#{1,6}\s"],  # Add regex for headings
#         keep_separator=True
#     )
#     return splitter.create_documents([text])

# import pdfplumber
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from typing import List

# def load_pdf(file_path: str) -> str:
#     """Load a PDF file and return the raw text."""
#     raw_text = ""
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 raw_text += text + "\n"       best
#     return raw_text

# def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
#     """Split text into structured, overlapping chunks for vector embedding."""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", ".", " ", r"^\s*[-•]\s+"],  # Handle bullet points
#         keep_separator=True
#     )
#     return splitter.create_documents([text])




import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

def load_pdf(file_path: str) -> str:
    """Load a PDF file and return the raw text."""
    raw_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
    return raw_text

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """Split text into structured, overlapping chunks for vector embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", r"^\s*[-•]\s+"],  # Handle lists
        keep_separator=True
    )
    return splitter.create_documents([text])