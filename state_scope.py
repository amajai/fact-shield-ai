import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    claim_statement: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add]
    notes: Annotated[list[str], operator.add] # notes ready for report generation
    final_report: str

class ClarifyClaim(BaseModel):
    """Schema for deciding whether the claim needs clarification before fact-checking."""
    need_clarification: bool = Field(
        description="True if clarification of the claim is required before fact-checking begins."
    )
    question: Optional[str] = Field(
        default=None,
        description="Clarifying question to ask the user about the claim, if needed."
    )
    verification: Optional[str] = Field(
        default=None,
        description="Message confirming fact-checking will begin after clarification."
    )

class FactCheckClaim(BaseModel):
    """Schema for representing the claim or statement under fact-checking."""
    claim_statement: str = Field(
        description="The exact claim or statement that will be fact-checked."
    )
