from agent_utils import RetrievalConfig, build_rag_agent, process_request
from index_utils import get_github_repo_docs, register_bank
from llama_stack_client import LlamaStackClient
from typing import List
import requests

## [P1] Given a repository, can I identify all the resources associated with it?

## [P0] Index documentation on separate memory bank
### recursive scrape of website

## [P0] Index code on separate memory bank
### github recursive thru repo

## [P0] Create agent associated with both memory banks

## [P1] Insert new document in appropriate memory bank


host = "localhost"
port = 5000
client = LlamaStackClient(base_url=f"http://{host}:{port}")
    
    
def insert_docs_in_bank(
    docs: List["Document"],
    bank: str,
    client = None
):  
    if client is None:
        client = LlamaStackClient(base_url=f"http://localhost:5000")
    
    existing_banks = [x.identifier for x in client.memory_banks.list()]
    if not bank in existing_banks:
        register_bank(client, bank)
        
    response = client.memory.insert(
        bank_id=bank,
        documents=docs
    )
    return response


def test_setup_index(owner="meta-llama", repo="llama-recipes"):
    extensions = ('py', 'md')
    docs = get_github_repo_docs(owner, repo, extensions)
    py = [d for d in docs if d['metadata']['extension'] == 'py']
    md = [d for d in docs if d['metadata']['extension'] == 'md']
    insert_docs_in_bank(py, f'{owner}/{repo}/python')
    insert_docs_in_bank(md, f'{owner}/{repo}/markdown')


def test_setup_agent_store(owner="meta-llama", repo="llama-recipes"):
    host = "localhost"
    port = 5000
    client = LlamaStackClient(base_url=f"http://{host}:{port}")
    
    # register an agent for each bank
    existing_banks = [x.identifier for x in client.memory_banks.list()]
    retrieval = RetrievalConfig(max_chunks=20, max_tokens_in_context=6000)
    agent_store = {bank: build_rag_agent(bank, retrieval, client) for bank in existing_banks}    
    return agent_store

    # TODO
    # register a master agent that can search the web. master consumes all the repsonses from agents and generates a new consolidated answer.
    
    

def test_query_agent(agent_store, query):
    host = "localhost"
    port = 5000
    client = LlamaStackClient(base_url=f"http://{host}:{port}")
    for bank, agent_id in agent_store.items():
        response = process_request(client, agent_id, query)
        print(f"Response from {bank}:\n==========\n")
        print(response['completion]'].content)
        print("Context used:\n==========\n")
        print(response['context_chunks'])
        
    return response

    
    