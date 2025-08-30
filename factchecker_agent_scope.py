import os
from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from prompts import clarify_fact_request_instructions, transform_messages_into_claim_prompt
from state_scope import AgentState, AgentInputState, ClarifyClaim, FactCheckClaim
from utils import get_today_str, create_llm
from dotenv import load_dotenv

load_dotenv()

model = create_llm() 

def clarify_fact_request(state: AgentState) -> Command[Literal["write_claim_statement", "__end__"]]:
    """
    Determine if the user's input statement contains enough factual information to begin fact-checking.

    Ensures deterministic, structured output:
    - If the input is a factual claim or statement, proceed to fact-checking brief generation.
    - If the input is a question, opinion, or insufficient for verification, return a clarification request instead.
    - Prevents hallucination or irrelevant analysis by strictly routing to either clarification or fact-check initiation.
    """
    structured_output_model = model.with_structured_output(ClarifyClaim)

    response = structured_output_model.invoke([
        HumanMessage(content=clarify_fact_request_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_claim_statement", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_claim_statement(state: AgentState):
    """
    Transform the conversation history into a comprehensive fact-checking brief.

    Ensures structured output by:
    - Converting user messages into a clear, detailed claim or statement for verification.
    - Including all explicitly stated details that support verification (dates, names, locations, figures, sources).
    - Explicitly noting any missing but important details as open considerations for clarification.
    - Guaranteeing the final brief follows the required format for effective fact-checking.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(FactCheckClaim)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_claim_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "claim_statement": response.claim_statement,
        "supervisor_messages": [HumanMessage(content=f"{response.claim_statement}.")]
    }

workflow = StateGraph(AgentState, input_schema=AgentInputState)

workflow.add_node("clarify_fact_request", clarify_fact_request)
workflow.add_node("write_claim_statement", write_claim_statement)

workflow.add_edge(START, "clarify_fact_request")
workflow.add_edge("write_claim_statement", END)

graph = workflow.compile()
