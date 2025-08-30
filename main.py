import asyncio
import os
import re
from datetime import datetime
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from utils import get_today_str, create_compress_llm
from prompts import final_report_generation_prompt
from state_scope import AgentState, AgentInputState
from factchecker_agent_scope import clarify_fact_request, write_claim_statement
from factchecker_multi_agent_supervisor import supervisor_agent
from niceterminalui import (
    print_banner, print_step, print_success, print_warning, 
    print_info, print_result_box, rich_prompt, print_completion_message,
    console
)


writer_model = create_compress_llm()


from state_scope import AgentState

async def final_report_generation(state: AgentState):
    """
    Final report generation node.
    
    Synthesizes all fact-checking research findings into a comprehensive verification report
    and saves it to the final_reports folder
    """
    
    notes = state.get("notes", [])
    
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("claim_statement", ""),
        findings=findings,
        date=get_today_str()
    )
    
    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    
    # Create final_reports directory if it doesn't exist
    os.makedirs("final_reports", exist_ok=True)
    
    # Extract title from the report (first line that starts with #)
    report_lines = final_report.content.split('\n')
    title = "fact_check_report"
    for line in report_lines:
        if line.strip().startswith('# '):
            title = line.strip()[2:]  # Remove "# " prefix
            break
    
    # Clean filename - remove invalid characters and limit length
    filename = re.sub(r'[^\w\s-]', '', title)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename[:50]  # Limit length
    
    filename = f"{filename}.md"
    
    # Save the report
    filepath = os.path.join("final_reports", filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_report.content)
    
    return {
        "final_report": final_report.content, 
        "report_filepath": filepath,
        "messages": [f"Fact-check completed! Final report saved to: {filepath}"],
    }

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_fact_request", clarify_fact_request)
deep_researcher_builder.add_node("write_claim_statement", write_claim_statement)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_fact_request")
deep_researcher_builder.add_edge("write_claim_statement", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# The clarify_fact_request node has conditional routing built-in via Command objects
# It will either go to "write_claim_statement" or END based on whether clarification is needed

agent = deep_researcher_builder.compile()


async def main():
    # Print beautiful banner
    print_banner(
        title="FactShield",
        subtitle="AI-Powered Fact-Checking System",
        description="Multi-Agent Verification & Analysis",
        subheader1="Comprehensive Claim Verification",
        subheader2="Evidence-Based Fact Checking"
    )
    
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}

    # Start with initial user input using rich prompt
    user_input = rich_prompt("What claim would you like me to fact-check?")
    messages = [HumanMessage(content=user_input)]
    
    while True:
        print_step("Starting Fact-Check Workflow", "🔍")
        
        current_state = {"messages": messages}
        final_state = None
        
        # Run the agent workflow
        async for event in agent.astream(current_state, config=thread):
            console.print(f"\n[dim]Processing: {list(event.keys())}[/dim]")
            for node, output in event.items():
                final_state = output
                if "messages" in output and output["messages"]:
                    latest_message = output["messages"][-1]
                    message_content = latest_message if isinstance(latest_message, str) else latest_message.content
                    
                    # Format different node outputs with appropriate styling
                    if node == "clarify_fact_request":
                        # Only show message if proceeding (not asking for clarification)
                        if not message_content.strip().endswith('?'):  # Simple check for questions
                            print_info(f"Claim Analysis: {message_content}")
                    elif node == "write_claim_statement":
                        print_success(f"Fact-Check Brief Generated")
                        if "claim_statement" in output:
                            print_info(f"Claim to verify: {output['claim_statement']}")
                    elif node == "supervisor_subgraph":
                        print_step("Fact-Checking In Progress", "🔬")
                    elif node == "final_report_generation":
                        if "saved to:" in message_content:
                            print_success(message_content)
                        else:
                            print_step("Generating Fact-Check Report", "📝")
        
        # Check if workflow ended at clarification step
        if final_state and len(final_state.get("messages", [])) > 0:
            last_message = final_state["messages"][-1]
            last_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # If workflow ended without final_report, it needs clarification
            if "final_report" not in final_state:
                print_warning("Need more information to verify this claim")
                console.print(f"\n[yellow]{last_content}[/yellow]")
                additional_info = rich_prompt("Please provide additional details")
                messages.append(HumanMessage(content=additional_info))
                continue
            else:
                # Display final report in a beautiful box
                if "final_report" in final_state:
                    print_result_box("Fact-Check Complete", final_state["final_report"])
                
                # Show completion message
                print_completion_message("FactShield", "Truth Through Evidence")
                break
        else:
            print_success("Fact-check workflow completed!")
            break

if __name__ == "__main__":
    asyncio.run(main())
    