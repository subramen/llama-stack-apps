# create agent
# associate banks with agent
# create session??

import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types import SamplingParams, UserMessage
from llama_stack_client.types.agent_create_params import (
    AgentConfig,
    AgentConfigToolMemoryToolDefinition,
)

from requests import session

"""
scratchpad
-----------

class AgentConfigToolMemoryToolDefinition(TypedDict, total=False):
    max_chunks: Required[int]
    max_tokens_in_context: Required[int]
    memory_bank_configs: MemoryBankConfig  ## Required[Iterable[AgentConfigToolMemoryToolDefinitionMemoryBankConfig]]
    query_generator_config: Required[AgentConfigToolMemoryToolDefinitionQueryGeneratorConfig]
    type: Required[Literal["memory"]]
    input_shields: List[str]
    output_shields: List[str]

class AgentConfig(TypedDict, total=False):
    enable_session_persistence: Required[bool]
    instructions: Required[str]
    max_infer_iters: Required[int]
    model: Required[str]
    input_shields: List[str]
    output_shields: List[str]
    sampling_params: SamplingParams
    tool_choice: Literal["auto", "required"]
    tool_prompt_format: Literal["json", "function_tag", "python_list"]
    "*3
    `json` -- Refers to the json format for calling tools. The json format takes the
    form like { "type": "function", "function" : { "name": "function_name",
    "description": "function_description", "parameters": {...} } }

    `function_tag` -- This is an example of how you could define your own user
    defined format for making tool calls. The function_tag format looks like this,
    <function=function_name>(parameters)</function>

    The detailed prompts for each of these formats are added to llama cli
    "*3
    tools: Iterable[AgentConfigTool]
"""


@dataclass
class RetrievalConfig:
    type: str = "memory"
    max_chunks: int = 10
    max_tokens_in_context: int = 2048
    input_shields: Optional[List[str]] = None
    output_shields: Optional[List[str]] = None


@dataclass
class MemoryBankConfig:
    bank_id: str
    type: str = "vector"


@dataclass
class QueryGenConfig:
    sep: str = "\n|||\n"


def get_rag_tool(
    retrieval_config: RetrievalConfig,
    memory_banks: List[str],
    query_generator_config: QueryGenConfig,
):
    rag_tool_config = asdict(retrieval_config)
    rag_tool_config["memory_bank_configs"] = [
        asdict(MemoryBankConfig(m)) for m in memory_banks
    ]
    rag_tool_config["query_generator_config"] = asdict(query_generator_config)
    return rag_tool_config


def build_agent(
    client: LlamaStackClient,
    model_id: Optional[str] = None,
    instructions: Optional[str] = None,
    tool_configs: List = [],
    sampling_params: SamplingParams = {},
    enable_session_persistence: bool = False,
    kwargs: Optional[dict] = {},
):
    if model_id is None:
        model_id = client.models.list()[0].identifier

    agent_config = AgentConfig(
        model=model_id,
        instructions=instructions,
        sampling_params=sampling_params,
        tools=tool_configs,
        enable_session_persistence=enable_session_persistence,
        **kwargs,
    )
    return Agent(client, agent_config)


# def query_agent(agent, user_prompt, session_id=None):
#     if session_id is None:
#         session_id = agent.create_session(f"test-{uuid.uuid4()}")

#     completion = agent.create_turn(
#         messages=[{"role": "user", "content": user_prompt}], session_id=session_id
#     )


def query_agent(client, agent, query, session_id=None):
    if session_id is None:
        session_id = agent.create_session(f"test-{uuid.uuid4()}")
        print(f"Created new session: {session_id}")

    agent_id = agent.agent_id
    messages = [{"role": "user", "content": query}]
    generator = client.agents.turn.create(
        agent_id=agent_id,
        messages=messages,
        stream=True,
    )

    turn = None
    ## Manually iterating here is a bit ugly;
    ## can this be under a function eg: generator.get_turn()
    for chunk in generator:
        event = chunk.event
        event_type = event.payload.event_type
        if event_type == "turn_complete":
            turn = event.payload.turn

    if turn is None:
        raise Exception("No turn returned")

    response = {
        "query": turn.input_messages[0].content,
        "context_chunks": turn.input_messages[0].context.split("\n|||\n"),
        "completion": turn.output_message,
        "session_id": session_id,
    }

    return response


#############


# def build_rag_agent(
#     memory_banks: List[str], retrieval_config: RetrievalConfig, client=None
# ):

#     if client is None:
#         client = LlamaStackClient(base_url=f"http://localhost:5000")

#     rag_tool_config = asdict(retrieval_config)
#     rag_tool_config["memory_bank_configs"] = [
#         asdict(MemoryBankConfig(m)) for m in memory_banks
#     ]
#     rag_tool_config["query_generator_config"] = asdict(QueryGenConfig())

#     model_id = client.models.list()[0].identifier

#     agent_config = AgentConfig(
#         model=model_id,
#         instructions="You are a helpful assistant. ",
#         sampling_params=SamplingParams(strategy="greedy", temperature=0.4, top_p=0.95),
#         tools=[rag_tool_config],
#         enable_session_persistence=True,
#     )
#     # response = client.agents.create(agent_config=agent_config)
#     # agent_id = response.agent_id
#     # return agent_id

#     return Agent(client, agent_config)


# def build_search_agent():
#     pass
