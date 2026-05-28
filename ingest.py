import os
os.environ["USE_TORCH"] = "1" # Force PyTorch, disable TensorFlow
import os
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

load_dotenv()

# Constants
REPO_ID = "Nikity/Epstein-Files"
DATA_DIR = os.getenv("DATA_PATH", "./data")
DB_DIR = os.getenv("DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def download_data(num_files=1):
    """Downloads a subset of the dataset from Hugging Face."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    downloaded_files = []
    for i in range(num_files):
        filename = f"epstein_files-{i:04d}.parquet"
        local_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=DATA_DIR
            )
        downloaded_files.append(local_path)
    return downloaded_files

def process_parquet(file_path):
    """Processes a parquet file and returns a list of LangChain Documents."""
    print(f"Processing {file_path}...")
    df = pd.read_parquet(file_path)
    
    documents = []
    # Based on research, the text column is likely 'text' or 'content'
    # We will check for available columns
    text_col = 'text_content' if 'text_content' in df.columns else ('text' if 'text' in df.columns else ('content' if 'content' in df.columns else None))
    
    if not text_col:
        print(f"Warning: No 'text' or 'content' column found in {file_path}. Columns: {df.columns}")
        return []

    for _, row in df.iterrows():
        text = str(row[text_col])
        if len(text.strip()) < 50: # Skip very short snippets
            continue
            
        metadata = {
            "source": os.path.basename(file_path),
            "original_filename": row.get('file_name', 'unknown')
        }
        documents.append(Document(page_content=text, metadata=metadata))
    
    return documents

def index_documents(documents):
    """Chunks documents and indices them into ChromaDB."""
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Indexing into ChromaDB at {DB_DIR}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("Indexing complete.")

if __name__ == "__main__":
    # For initial run, just download 1 file (~500MB likely)
    files = download_data(num_files=1)
    
    all_docs = []
    for f in files:
        docs = process_parquet(f)
        all_docs.extend(docs)
    
    if all_docs:
        index_documents(all_docs)
    else:
        print("No documents found to index.")
