# create agent
# associate banks with agent
# create session??

import uuid
from dataclasses import asdict, dataclass
from typing import List, Optional


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


def build_rag_agent(
    memory_banks: List[str], retrieval_config: RetrievalConfig, client=None
):
    pass


def build_search_agent():
    pass


def process_request(client, agent_id, query):
    pass
