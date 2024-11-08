import os
from typing import List

from agent_utils import (
    build_agent,
    get_rag_tool,
    query_agent,
    QueryGenConfig,
    RetrievalConfig,
)
from index_utils import create_index, get_directory_files, get_github_repo_docs
from llama_stack_client import LlamaStackClient

from requests import session

## [P1] Given a repository, can I identify all the resources associated with it?

## [P0] Index documentation on separate memory bank
### recursive scrape of website

## [P0] Index code on separate memory bank
### github recursive thru repo

## [P0] Create agent associated with both memory banks

## [P1] Insert new document in appropriate memory bank


host = "localhost"
port = 5000
CLIENT = LlamaStackClient(base_url=f"http://{host}:{port}")


def test_setup_index_files(directory):
    extensions = ("py", "md")
    docs = get_directory_files(directory, extensions)
    py = [d for d in docs if d["metadata"]["extension"] == "py"]
    md = [d for d in docs if d["metadata"]["extension"] == "md"]
    create_index(CLIENT, py, f"{directory}/python")
    create_index(CLIENT, md, f"{directory}/markdown")


def test_setup_index_gh(owner, repo):
    extensions = ("py", "md")
    docs = get_github_repo_docs(owner, repo, extensions)
    py = [d for d in docs if d["metadata"]["extension"] == "py"]
    md = [d for d in docs if d["metadata"]["extension"] == "md"]
    create_index(CLIENT, py, f"{owner}/{repo}/python")
    create_index(CLIENT, md, f"{owner}/{repo}/markdown")


def test_setup_rag_ensemble():
    # register an agent for each bank
    memory_banks = [x.identifier for x in CLIENT.memory_banks.list()]
    retrieval_config = RetrievalConfig(max_chunks=20, max_tokens_in_context=6000)

    # need more details on what this is and how to use it
    # can this include the prompt template i.e. how the context_str and query_str are passed to the llm?
    query_gen_config = QueryGenConfig()

    rag_ensemble = {}
    for bank in memory_banks:
        file_type = bank.split("/")[-1]
        system_prompt = f"""
        You are an experienced software developer and an expert at answering questions about software libraries. 
        Given a question, use the provided context obtained from this library's {file_type} files to answer the question. 
        """
        rag_tool = get_rag_tool(retrieval_config, [bank], query_gen_config)
        agent = build_agent(
            CLIENT,
            instructions=system_prompt,
            tool_configs=[rag_tool],
            sampling_params={"top_p": 0.9, "temperature": 0.7},
        )
        rag_ensemble[bank] = agent

    return rag_ensemble


def test_setup_search_agent():
    system_prompt = "You are an expert customer support agent for open source software libraries. You are helped by a team of specialists who provide you with the context you need to answer questions. Use this context to synthesize a high-quality answer to the question. You can search the web to verify your answer or include additional information if necessary."

    search_tool = {
        "type": "brave_search",
        "engine": "brave",
        "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
    }

    agent = build_agent(
        CLIENT,
        instructions=system_prompt,
        tool_configs=[search_tool],
        sampling_params={"top_p": 0.9, "temperature": 0.7},
        # not sure what these do exactly
        kwargs=dict(
            tool_choice="auto",
            tool_prompt_format="function_tag",
        ),
    )
    return agent


def test_query_ensemble(rag_ensemble, query):
    ensemble_responses = {}
    session_id = None
    for name, agent in rag_ensemble.items():
        response = query_agent(CLIENT, agent, query, session_id)
        session_id = response["session_id"]
        ensemble_responses[name] = response
    return ensemble_responses
