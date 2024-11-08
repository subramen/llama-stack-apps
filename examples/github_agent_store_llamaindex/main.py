from agent_utils import *
from index_utils import *

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.groq import Groq


## [P1] Given a repository, can I identify all the resources associated with it?

## [P0] Index documentation on separate memory bank
### recursive scrape of website

## [P0] Index code on separate memory bank
### github recursive thru repo

## [P0] Create agent associated with both memory banks

## [P1] Insert new document in appropriate memory bank


def test_setup_index_files(directory):
    extensions = ("py", "md")
    docs = get_directory_files(directory, extensions)
    py = [d for d in docs if d.metadata["extension"] == "py"]
    md = [d for d in docs if d.metadata["extension"] == "md"]
    return {
        "py": create_index(py, persist=True),
        "md": create_index(md, persist=True),
    }


def test_setup_index_gh(owner, repo):
    extensions = ("py", "md")
    docs = get_github_repo_files(owner, repo, extensions)
    py = [d for d in docs if d.metadata["extension"] == "py"]
    md = [d for d in docs if d.metadata["extension"] == "md"]
    return {
        "py": create_index(py, persist=True),
        "md": create_index(md, persist=True),
    }


def test_setup_rag_ensemble(indexes):
    qa_prompt_tmpl = (
        "Here is some context information gathered from an open-source repo's {filetype} files:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Please also write the answer in the tone of {tone_name}.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl)

    rag_ensemble = {}
    for name, index in indexes.items():
        retriever = VectorIndexRetriever(index=index, similarity_top_k=10, filters=None)
        synthesizer = TreeSummarize(
            llm=Groq(
                model="llama-3.2-3b-preview",
                api_key="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",
            ),
            summary_template=qa_prompt,
            verbose=True,
        )
        rag_ensemble[name] = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=synthesizer
        )
    return rag_ensemble


def test_setup_search_agent():
    raise NotImplementedError


def test_query_ensemble(agent_store, query):
    ensemble_responses = {}
    for name, agent in agent_store.items():
        print(f"Querying {name}...\n===========\n")
        knn_chunks = agent.retrieve(query)  # ~ client.memory.query(query)
        chunk_str = [n.get_text() for n in knn_chunks]
        answer = agent._response_synthesizer.get_response(
            query, chunk_str, tone_name="a professional software developer"
        )
        ensemble_responses[name] = {
            "query": query,
            "completion": answer,
            "context_chunks": knn_chunks,
        }
    return ensemble_responses


def test_main():
    directory = "/Users/subramen/GITHUB/llama-recipes/recipes/quickstart"
    indexes = test_setup_index_files(directory)
    agents = test_setup_agent_store(indexes)
    query = "What are all the things I can do with llama?"
