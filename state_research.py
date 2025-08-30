import operator
from typing_extensions import TypedDict, Annotated, List, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class FactCheckerState(TypedDict):
    """
    State for the fact-checking agent containing conversation history and metadata.

    Tracks the claim being fact-checked, iteration count for limiting tool calls,
    compressed evidence summaries, and raw evidence notes for detailed analysis.
    """
    tool_call_iterations: int
    claim_statement: Optional[str]
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    fact_checker_messages: Annotated[Sequence[BaseMessage], add_messages]

class FactCheckerOutputState(TypedDict):
    """
    Output state for the fact-checking agent containing final results.

    Represents the final output of the fact-checking process with compressed
    evidence summaries and all raw evidence notes from the process.
    """
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    fact_checker_messages: Annotated[Sequence[BaseMessage], add_messages]

class EvidenceSummary(BaseModel):
    """Schema for summarizing webpage evidence related to a claim, including stance analysis."""
    
    summary: str = Field(
        description="Concise summary of the webpage evidence"
    )
    
    key_excerpts: List[str] = Field(
        description="Important quotes or excerpts from the content, up to 5 items"
    )
    
    stance: str = Field(
        description="The stance of this evidence toward the claim. One of: 'Supports', 'Contradicts', 'Mixed', or 'Unclear'."
    )
