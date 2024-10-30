# create agent
# associate banks with agent
# create session??

from typing import Optional, List
from dataclasses import dataclass

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

    
def build_rag_agent(
    client,
    agent_name:str, 
    memory_banks:List[str], 
    retrieval_config: RetrievalConfig, 
    model: str):
    
    rag_tool_config = AgentConfigToolMemoryToolDefinition(**retrieval_config)
    rag_tool_config['memory_bank_configs'] = [asdict(MemoryBankConfig(m)) for m in memory_banks]
    
    agent_config = AgentConfig(
        model=model,
        instructions="You are a helpful assistant. ",
        sampling_params=SamplingParams(
            strategy="greedy", temperature=0.4, top_p=0.95
        ),
        tools=[rag_tool_config],
        enable_session_persistence=True,
    )
    
    response = client.agents.create(agent_config=agent_config)
    agent_id = response.agent_id
    return agent_id
    

def format_prompt(rag_query):
    user_msg = UserMessage(content=)

def process_request(client, agent_id, query):
    response = client.agents.session.create(
        agent_id=agent_id,
        session_name=f"Session-{uuid.uuid4()}",
    )
    session_id = response.session_id
    