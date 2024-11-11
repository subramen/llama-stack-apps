import os
from typing import List

from agent_utils import (
    build_agent,
    get_rag_tool,
    query_agent,
    QueryGenConfig,
    RetrievalConfig,
)
from index_utils import create_index, get_directory_docs, get_github_repo_docs
from llama_stack_client import LlamaStackClient

from dotenv import load_dotenv
load_dotenv()

host = "localhost"
port = 5000
CLIENT = LlamaStackClient(base_url=f"http://{host}:{port}")


def test_setup_index_files(directory):
    extensions = ("py", "md")
    docs = get_directory_docs(directory, extensions)
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
    query_gen_config = QueryGenConfig()  # no-op at this time

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
        "api_key": os.getenv("BRAVE_SEARCH_API_KEY")
    }

    agent = build_agent(
        CLIENT,
        instructions=system_prompt,
        tool_configs=[search_tool],
        sampling_params={"top_p": 0.9, "temperature": 0.7},
        kwargs=dict(
            tool_choice="auto",
            tool_prompt_format="function_tag",
        ),
    )
    return agent


def test_query_ensemble(rag_ensemble, query):
    ensemble_responses = {}
    messages = [{"role": "user", "content": query}]
    for name, agent in rag_ensemble.items():
        response = query_agent(CLIENT, agent, messages)
        ensemble_responses[name] = response
    return ensemble_responses


def test():
    directory = "/home/ubuntu/subramen/llama-recipes"
    query = "What methods are best for finetuning llama?"
    
    test_setup_index_files(directory)
    rag_store = test_setup_rag_ensemble()
    answers = test_query_ensemble(rag_store, query)
    
    
    context = '\n* '.join([a['completion'].content for a in answers.values()])
    message = [{
        "role": "user",
        "content": f"Query: {query}\n\nSpecialist answers:{context}"
    }]
    
    try:
        # This doesn't work https://github.com/meta-llama/llama-stack/issues/407
        search_agent = test_setup_search_agent()
        response = query_agent(CLIENT, search_agent, message)
        print(response['completion'].content)
    except:
        print(message[0]['content'])
    
    ### role='user' content='Query: What methods are best for finetuning llama?\n\nSpecialist answers:Based on the provided context, it appears that finetuning LLaMA is not directly mentioned in the code snippets. However, I can infer that finetuning LLaMA is likely to be performed using the `llama_recipes.finetuning` module.\n\nIn the `finetuning.py` file, the `main` function is imported from `llama_recipes.finetuning`, which suggests that this file contains the code for finetuning LLaMA.\n\nTo finetun...<more>...Guard.\n\nAs for finetuning Llama in general, it seems that the provided context only provides information on finetuning Llama Guard, which is a specific application of the Llama model. For general finetuning of Llama, you may need to refer to the official documentation or other external resources.\n\nHowever, based on the provided context, it seems that the `finetune_vision_model.md` file in the `quickstart` folder may provide some information on finetuning Llama for vision tasks.' context=None
    ### AssertionError: Tool code_interpreter not found
    ### How to disable using code_interpreter?
    
if __name__ == "__main__":
    test()
    