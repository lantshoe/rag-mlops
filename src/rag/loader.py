from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

def load_documents(file_path: str):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()
    return documents

def split_documents(documents, chunk_size=512, chunk_overlap=50):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
