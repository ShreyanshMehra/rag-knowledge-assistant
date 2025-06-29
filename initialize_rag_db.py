import os
from dotenv import load_dotenv, find_dotenv
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

_ = load_dotenv(find_dotenv())

CHROMA_PATH = "chroma"
DATA_PATH = "data/aws_ec2_documentation"

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    # Use TextLoader with autodetect_encoding to avoid UnicodeDecodeError
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs=text_loader_kwargs
    )
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,      
        chunk_overlap=128,    
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if len(chunks) > 20:
        document = chunks[20]
        print(document.page_content)
        print(document.metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}   # Use GPU
    )
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Saved {len(chunks)} chunks to database in {CHROMA_PATH}.")

if __name__ == "__main__":
    generate_data_store()
