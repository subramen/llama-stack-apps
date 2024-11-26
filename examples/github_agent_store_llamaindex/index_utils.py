import logging
import os
from typing import Callable, List, Optional, Tuple

import fire
import requests

from llama_index.core import Document, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from smart_open import open as smart_open
from tenacity import before_sleep_log, retry, wait_exponential
from typing_extensions import Doc


"""
IDEAS SCRATCHPAD

- FolderBank class that encapsulates the root dir, filetypes, and the various banks related to the files in that folder

"""
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def read(filepath: str):
    with smart_open(filepath) as f:
        return f.read()


def get_github_repo_files(
    owner: str,
    repo: str,
    extensions: Optional[Tuple[str]] = None,
) -> List[Document]:
    base_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if api_key:
        headers["Authorization"] = f"token {os.getenv("GITHUB_API_KEY")}"
    docs = []

    @retry(
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def fetch_files(url):
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def process_file_to_doc(file_info):
        if file_info["type"] == "file":
            file_extension = file_info["name"].split(".")[-1]
            if extensions is None or file_extension in extensions:
                raw_url = file_info["download_url"]
                raw_content = requests.get(raw_url).text
                docs.append(
                    Document(
                        text=raw_content,
                        metadata={
                            "rel_path": file_info["path"],
                            "extension": file_extension,
                            "html_url": file_info["_links"]["html"],
                        },
                        doc_id=raw_url,
                    )
                )
        elif file_info["type"] == "dir":
            subdir_url = file_info["_links"]["self"]
            subdir_files = fetch_files(subdir_url)
            for sub_file_info in subdir_files:
                process_file_to_doc(sub_file_info)

    root_files = fetch_files(base_url)
    for file_info in root_files:
        process_file_to_doc(file_info)
    return docs


def get_directory_files(
    directory: str, extensions: Optional[Tuple[str]] = None
) -> List[Document]:
    docs = []

    def fetch_files(directory, extensions: Optional[Tuple[str]]):
        paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
        ]
        if extensions:
            paths = [p for p in paths if p.split(".")[-1] in extensions]
        return paths

    def process_file_to_doc(filepath):
        content = read(filepath)
        docs.append(
            Document(
                text=content,
                metadata={
                    "root_dir": directory,
                    "relpath": os.path.relpath(filepath, directory),
                    "extension": filepath.split(".")[-1],
                },
                doc_id=filepath,
            )
        )

    files = fetch_files(directory, extensions)
    for file in files:
        process_file_to_doc(file)
    return docs


def create_indexes(docs: List[Document], persist=True):
    # index = VectorStoreIndex.from_documents(docs)  # equivalent to memory_banks.register() + memory.insert()
    splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separator=" ",
    )
    nodes = splitter.get_nodes_from_documents(docs)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    summary_index = SummaryIndex(nodes, embed_model=embed_model)
    if persist:
        vector_index.storage_context.persist(persist_dir="./cache")
        summary_index.storage_context.persist(persist_dir="./cache")

    return {
        "vector_index": vector_index,
        "summary_index": summary_index,
    }
