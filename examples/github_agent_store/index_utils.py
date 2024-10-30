import os
from typing import Optional
"""
IDEAS SCRATCHPAD

- FolderBank class that encapsulates the root dir, filetypes, and the various banks related to the files in that folder

"""


def register_new_bank(
    client,
    identifier,
    embedding_model="all-MiniLM-L6-v2",
    chunk_size_in_tokens=512,
    overlap_size_in_tokens=64
):
    providers = client.providers.list()
    response = client.memory_bank.register(
        memory_bank=dict(
            identifier=identifier,
            embedding_model=embedding_model,
            chunk_size_in_tokens=chunk_size_in_tokens,
            overlap_size_in_tokens=overlap_size_in_tokens,
            provider_id=providers["memory"][0].provider_id
        )
    )
    return response


def listdir_recursive(root, file_type: Optional[str]):
    paths = [
        os.path.join(root, file)  
        for root, _, files in os.walk(directory) 
        for file in files
    ]
    if file_type:
        paths = [p for p in paths if p[-3:] == file_type]
    return paths


def read(filepath: str):
    with open(filepath) as f:
        return f.read()


def add_dir_to_bank(
    bank_identifier: str,
    file_dir: str,
    host: str = "localhost", 
    port: int = 5000,
    file_type: Optional[str] = None,
    metadata_gen: Optional[Callable] = None
):
    client = LlamaStackClient(base_url=f"http://{host}:{port}")
    files = listdir_recursive(file_dir, file_type)
    docs = [
        Document(
            document_id=file,
            content=read(file),
            mime_type="text/plain",
            metadata=metadata_gen(file) or {},
        )
        for file in files
    ]
    response = client.memory.insert(
        bank_id=bank_identifier,
        documents=docs
    )
    return response


if __name__ == "__main__":
    fire.Fire(add_dir_to_bank)