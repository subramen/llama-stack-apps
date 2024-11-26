import os

from dotenv import load_dotenv
from index_utils import create_indexes, get_directory_files, get_github_repo_files
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.groq import Groq

load_dotenv()


def test_setup_index_files(directory):
    extensions = ("py", "md")
    docs = get_directory_files(directory, extensions)
    py = [d for d in docs if d.metadata["extension"] == "py"]
    md = [d for d in docs if d.metadata["extension"] == "md"]
    return {
        "py": create_indexes(py, persist=True),
        "md": create_indexes(md, persist=True),
    }


def test_setup_index_gh(owner, repo):
    extensions = ("py", "md")
    docs = get_github_repo_files(owner, repo, extensions)
    py = [d for d in docs if d.metadata["extension"] == "py"]
    md = [d for d in docs if d.metadata["extension"] == "md"]
    return {
        "py": create_indexes(py, persist=True),
        "md": create_indexes(md, persist=True),
    }


def test_setup_rag_ensemble(indexes):
    groq_llm = Groq(model="llama-3.2-3b-preview", api_key=os.getenv("GROQ_API_KEY"))

    qa_prompt_tmpl = (
        "Here is some context information gathered from an open-source repo's {filetype} files:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query.\n"
        "Please also write the answer in the tone of a professional software developer.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl)

    rag_ensemble = {}
    for name, indexes in indexes.items():
        summary_index = indexes["summary_index"]
        summary_query_engine = summary_index.as_query_engine(llm=groq_llm)

        vector_index = indexes["vector_index"]
        retriever = vector_index.as_retriever(similarity_top_k=10)
        synthesizer = TreeSummarize(
            llm=groq_llm,
            summary_template=qa_prompt,
            verbose=True,
        )
        vector_query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=synthesizer
        )

        rag_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" {name} documents."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        f" of EVERYTHING about the {name} type document. For questions about"
                        " more specific sections, please use the vector_tool."
                    ),
                ),
            ),
        ]

        agent = ReActAgent(tools=rag_tools, llm=groq_llm)

    return rag_ensemble


def test_setup_search_agent():
    raise NotImplementedError


def test_query_ensemble(agent_store, query):
    ensemble_responses = {}
    for name, agent in agent_store.items():
        print(f"Querying {name}...\n===========\n")
        knn_chunks = agent.retrieve(query)  # ~ client.memory.query(query)
        answer = agent.synthesize(query, knn_chunks)

        #### low-level API to pass addl args to custom_template ####
        # chunk_str = [n.get_text() for n in knn_chunks]
        # answer = agent._response_synthesizer.get_response(
        #     query, chunk_str, tone_name="a professional software developer"
        # )

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
