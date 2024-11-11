import logging
import os
from typing import Callable, List, Optional, Tuple

import fire
import requests

from llama_stack_client import LlamaStackClient
from llama_stack_client.types.memory_insert_params import Document
from smart_open import open as smart_open
from tenacity import before_sleep_log, retry, wait_exponential


"""
IDEAS SCRATCHPAD

- FolderBank class that encapsulates the root dir, filetypes, and the various banks related to the files in that folder

"""
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def read(filepath: str):
    with smart_open(filepath) as f:
        return f.read()


def get_github_repo_docs(
    owner: str,
    repo: str,
    extensions: Optional[Tuple[str]] = None,
):
    base_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    headers = {"Accept": "application/vnd.github.v3+json"}
    headers["Authorization"] = f"token {os.getenv('GITHUB_API_KEY')}"
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
                        document_id=raw_url,
                        content=raw_content,
                        mime_type="text/plain",
                        metadata={
                            "rel_path": file_info["path"],
                            "extension": file_extension,
                            "html_url": file_info["_links"]["html"],
                        },
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


def get_directory_docs(directory: str, extensions: Optional[Tuple[str]] = None):
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
                document_id=filepath,
                content=content,
                mime_type="text/plain",
                metadata={
                    "root_dir": directory,
                    "relpath": os.path.relpath(filepath, directory),
                    "extension": filepath.split(".")[-1],
                },
            )
        )

    files = fetch_files(directory, extensions)
    for file in files:
        process_file_to_doc(file)
    return docs


def register_bank(
    client,
    identifier,
    embedding_model="all-MiniLM-L6-v2",
    chunk_size_in_tokens=512,
    overlap_size_in_tokens=64,
):
    providers = client.providers.list()
    response = client.memory_banks.register(
        memory_bank=dict(
            identifier=identifier,
            embedding_model=embedding_model,
            chunk_size_in_tokens=chunk_size_in_tokens,
            overlap_size_in_tokens=overlap_size_in_tokens,
            provider_id=providers["memory"][0].provider_id,
        )
    )
    return response


def create_index(
    client: LlamaStackClient,
    docs: List[Document],
    bank: str,
):
    existing_banks = [x.identifier for x in client.memory_banks.list()]
    if not bank in existing_banks:
        register_bank(client, bank)

    # API should return the error/success code
    response = client.memory.insert(bank_id=bank, documents=docs)
    # return response.status_code
