import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types import SamplingParams, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig

from requests import session


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
    type: str = "default"     # what does this do?
    sep: str = "\n|||\n"     # what does this do?


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


def query_agent(client, agent, messages, session_id=None):
    if session_id is None:
        session_id = agent.create_session(f"test-{uuid.uuid4()}")
        print(f"Created new session: {session_id}")

    agent_id = agent.agent_id
    generator = client.agents.turn.create(
        agent_id=agent_id,
        messages=messages,
        session_id=session_id,
        stream=True,
    )

    turn = None
    for chunk in generator:
        event = chunk.event
        event_type = event.payload.event_type
        if event_type == "turn_complete":
            turn = event.payload.turn

    if turn is None:
        raise Exception("No turn returned")

    response = {
        "query": turn.input_messages[0].content,
        "context_chunks": turn.input_messages[0].context,
        "completion": turn.output_message,
        "session_id": session_id,
    }

    return response
