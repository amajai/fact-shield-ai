import asyncio
import time
from datetime import datetime
from typing import Dict, Any

import streamlit as st
from langchain_core.messages import HumanMessage

# Import the workflow components from main.py
from main import agent

# Configure Streamlit page
st.set_page_config(
    page_title="FactShield - AI Fact Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .workflow-step {
        padding: 1rem;
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc;
        margin: 1rem 0;
        border-radius: 5px;
        color: #0f172a; /* slate-900 for readability on light bg */
    }
    
    .success-box {
        padding: 1rem;
        background-color: #dcfce7;
        border: 1px solid #22c55e;
        border-radius: 5px;
        margin: 1rem 0;
        color: #065f46; /* emerald-800 */
    }
    
    .warning-box {
        padding: 1rem;
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 5px;
        margin: 1rem 0;
        color: #78350f; /* amber-900 */
    }
    
    .error-box {
        padding: 1rem;
        background-color: #fee2e2;
        border: 1px solid #ef4444;
        border-radius: 5px;
        margin: 1rem 0;
        color: #7f1d1d; /* rose-900 */
    }

    /* Timestamps and links inside workflow items */
    .workflow-step small, .success-box small, .warning-box small, .error-box small {
        color: #64748b; /* slate-500 */
    }
    .workflow-step a, .success-box a, .warning-box a, .error-box a {
        color: #1d4ed8; /* blue-700 for links */
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Display the main header for the application."""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ FactShield</h1>
        <h3>AI-Powered Fact-Checking System</h3>
        <p>Multi-Agent Verification & Analysis | Evidence-Based Fact Checking</p>
    </div>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = 'input'
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'final_report' not in st.session_state:
        st.session_state.final_report = None
    if 'claim_statement' not in st.session_state:
        st.session_state.claim_statement = None
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def add_workflow_step(step_name: str, status: str, message: str):
    """Add a step to the workflow history."""
    st.session_state.workflow_history.append({
        'step': step_name,
        'status': status,  # 'processing', 'completed', 'warning', 'error'
        'message': message,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })

def display_workflow_history():
    """Display the workflow progress history."""
    if st.session_state.workflow_history:
        st.subheader("📋 Workflow Progress")
        
    # Show newest first
    for step in reversed(st.session_state.workflow_history):
            status_emoji = {
                'processing': '⏳',
                'completed': '✅',
                'warning': '⚠️',
                'error': '❌'
            }.get(step['status'], '📝')
            
            status_class = {
                'processing': 'workflow-step',
                'completed': 'success-box',
                'warning': 'warning-box',
                'error': 'error-box'
            }.get(step['status'], 'workflow-step')
            
            st.markdown(f"""
            <div class="{status_class}">
                <strong>{status_emoji} {step['step']}</strong> <small>({step['timestamp']})</small><br>
                {step['message']}
            </div>
            """, unsafe_allow_html=True)

async def run_fact_check_workflow(messages_list: list, step_container, progress_container) -> Dict[str, Any]:
    """
    Run the fact-checking workflow asynchronously.
    
    Args:
        messages_list: List of messages (strings) from the conversation
        step_container: Streamlit container for step updates
        progress_container: Streamlit container for progress messages
        
    Returns:
        Dictionary containing the final workflow state
    """
    thread = {"configurable": {"thread_id": "streamlit_session", "recursion_limit": 50}}
    # Convert string messages to HumanMessage objects - matching main.py pattern
    messages = [HumanMessage(content=msg) for msg in messages_list]
    current_state = {"messages": messages}
    final_state = None
    
    try:
        add_workflow_step("Workflow Started", "processing", "Starting fact-check workflow...")
        progress_container.info("🚀 Starting fact-check workflow...")
        
        # Update display
        with step_container.container():
            display_workflow_history()
        
        # Run the agent workflow - mirroring main.py logic
        async for event in agent.astream(current_state, config=thread):
            # Show which node is being processed
            node_names = list(event.keys())
            if node_names:
                add_workflow_step("Processing", "processing", f"Running: {', '.join(node_names)}")
                progress_container.info(f"⚙️ Processing: {', '.join(node_names)}")
                
                # Update display immediately to show progress
                with step_container.container():
                    display_workflow_history()
            
            for node, output in event.items():
                final_state = output
                
                if "messages" in output and output["messages"]:
                    latest_message = output["messages"][-1]
                    message_content = latest_message if isinstance(latest_message, str) else latest_message.content
                    
                    # Format different node outputs - matching main.py logic
                    if node == "clarify_fact_request":
                        # Show progress for this node
                        add_workflow_step("Analyzing Claim", "processing", "Examining the claim for clarity and specificity...")
                        progress_container.info("🔍 Analyzing claim for clarity...")
                        
                        # Check if this is asking for clarification or proceeding
                        is_clarification = (
                            message_content.strip().endswith('?') or
                            'need more' in message_content.lower() or
                            'please provide' in message_content.lower() or
                            'clarify' in message_content.lower() or
                            'specify' in message_content.lower() or
                            'additional' in message_content.lower()
                        )
                        
                        if not is_clarification:
                            add_workflow_step("Claim Analysis", "completed", f"Analysis: {message_content}")
                            progress_container.success("✅ Claim analysis completed")
                        else:
                            add_workflow_step("Clarification Request", "warning", "System requesting more information...")
                            progress_container.warning("⚠️ More information needed...")
                    elif node == "write_claim_statement":
                        add_workflow_step("Generating Brief", "processing", "Creating structured fact-check brief...")
                        progress_container.info("📝 Generating fact-check brief...")
                        
                        add_workflow_step("Claim Statement", "completed", "Fact-Check Brief Generated")
                        if "claim_statement" in output:
                            st.session_state.claim_statement = output["claim_statement"]
                            add_workflow_step("Claim Extraction", "completed", f"Claim to verify: {output['claim_statement']}")
                        progress_container.success("✅ Fact-check brief generated")
                    elif node == "supervisor_subgraph":
                        add_workflow_step("Research Phase", "processing", "Fact-checking in progress...")
                        progress_container.info("🔬 Fact-checking in progress...")
                    elif node == "final_report_generation":
                        if "messages" in output and output["messages"]:
                            latest_msg = output["messages"][-1]
                            msg_content = latest_msg if isinstance(latest_msg, str) else latest_msg.content
                            
                            if "saved to:" in msg_content:
                                add_workflow_step("Report Generation", "completed", msg_content)
                                progress_container.success("✅ Report generated and saved")
                            else:
                                add_workflow_step("Report Generation", "processing", "Generating fact-check report...")
                                progress_container.info("📝 Generating fact-check report...")
                
                # Update display in real-time
                with step_container.container():
                    display_workflow_history()
        
        # Critical logic from main.py lines 141-163: Check if workflow ended at clarification step
        if final_state and len(final_state.get("messages", [])) > 0:
            last_message = final_state["messages"][-1]
            last_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # If workflow ended without final_report, it needs clarification - EXACT logic from main.py
            if "final_report" not in final_state:
                add_workflow_step("Clarification Needed", "warning", f"System needs more information: {last_content}")
                progress_container.warning("⚠️ Additional information required to complete fact-check")
                return {"needs_clarification": True, "message": last_content, "final_state": final_state}
            else:
                # Workflow completed successfully
                if "final_report" in final_state:
                    add_workflow_step("Workflow Complete", "completed", "Fact-check completed successfully")
                    progress_container.success("✅ Fact-check workflow completed!")
                    return {"completed": True, "final_report": final_state["final_report"], "final_state": final_state}
        else:
            add_workflow_step("Workflow Complete", "completed", "Fact-check workflow completed")
            progress_container.success("✅ Fact-check workflow completed!")
            return {"completed": True, "final_state": final_state}
        
    except Exception as e:
        add_workflow_step("Error", "error", f"Workflow failed: {str(e)}")
        progress_container.error(f"❌ An error occurred: {str(e)}")
        return {"error": str(e)}

def main():
    """Main Streamlit application."""
    display_header()
    initialize_session_state()
    
    # Sidebar with information
    with st.sidebar:
        st.header("🔍 How FactShield Works")
        st.markdown("""
        1. **Input Analysis**: Submit a claim for verification
        2. **Claim Processing**: AI analyzes and structures your claim
        3. **Multi-Agent Research**: Specialized agents conduct comprehensive fact-checking
        4. **Report Generation**: Generate detailed verification report with evidence
        """)
        
        st.header("📊 Session Info")
        st.write(f"Workflow State: `{st.session_state.workflow_state}`")
        st.write(f"Messages: {len(st.session_state.messages)}")
        
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        if st.session_state.workflow_state == 'input' and not st.session_state.processing:
            st.subheader("🎯 Submit Claim for Fact-Checking")
            
            claim_input = st.text_area(
                "What claim would you like me to fact-check?",
                placeholder="Enter a specific claim, statement, or allegation that you want verified...",
                height=150,
                key="claim_input"
            )
            
            if st.button("🔍 Start Fact-Check", type="primary", disabled=not claim_input.strip()):
                if claim_input.strip():
                    st.session_state.processing = True
                    st.session_state.workflow_state = 'processing'
                    st.session_state.messages = [claim_input]
                    st.session_state.workflow_history = []
                    st.rerun()
        
        # Processing section
        elif st.session_state.workflow_state == 'processing':
            st.subheader("⚙️ Processing Your Claim")
            
            # Display current claim
            with st.expander("📝 Current Claim", expanded=True):
                st.write(st.session_state.messages[-1])  # Show the most recent message
            
            # Create containers for real-time updates
            progress_container = st.empty()
            step_container = st.empty()
            
            # Run the workflow
            if st.session_state.processing:
                # Run async workflow
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # Pass the containers to the workflow function for real-time updates
                    result = loop.run_until_complete(
                        run_fact_check_workflow(
                            st.session_state.messages,  # Pass all messages 
                            step_container, 
                            progress_container
                        )
                    )
                    
                    # Handle different result types based on main.py logic
                    if result.get("needs_clarification"):
                        st.session_state.workflow_state = 'clarification'
                        st.session_state.clarification_message = result.get("message", "")
                        progress_container.warning("⚠️ Additional information needed")
                    elif result.get("completed") and result.get("final_report"):
                        st.session_state.final_report = result["final_report"]
                        st.session_state.workflow_state = 'completed'
                        progress_container.success("✅ Fact-check completed!")
                    elif result.get("completed"):
                        # Workflow completed but without final report (edge case)
                        st.session_state.workflow_state = 'error'
                        add_workflow_step("Warning", "warning", "Workflow completed without generating a report")
                        progress_container.warning("⚠️ Workflow completed but no report generated")
                    elif result.get("error"):
                        st.session_state.workflow_state = 'error'
                        progress_container.error(f"❌ Error: {result.get('error')}")
                    else:
                        st.session_state.workflow_state = 'error'
                        add_workflow_step("Error", "error", "Unexpected workflow completion")
                        progress_container.error("❌ Unexpected workflow result")
                    
                    st.session_state.processing = False
                    
                except Exception as e:
                    st.session_state.workflow_state = 'error'
                    st.session_state.processing = False
                    add_workflow_step("Error", "error", f"Exception during workflow: {str(e)}")
                    progress_container.error(f"❌ Exception: {str(e)}")
                        
                finally:
                    loop.close()
                
                # Auto-refresh to move to next state
                time.sleep(1)  # Brief pause to show final status
                st.rerun()
        
        # Clarification needed section
        elif st.session_state.workflow_state == 'clarification':
            st.subheader("❓ Additional Information Needed")
            
            st.warning("The system needs more information to properly fact-check your claim.")
            
            # Show the clarification message from the workflow
            if hasattr(st.session_state, 'clarification_message') and st.session_state.clarification_message:
                st.info(st.session_state.clarification_message)
            
            # Show the conversation so far
            st.subheader("📝 Conversation History")
            for i, msg in enumerate(st.session_state.messages):
                if i == 0:
                    st.write(f"**Original Claim:** {msg}")
                else:
                    st.write(f"**Additional Info {i}:** {msg}")
            
            additional_info = st.text_area(
                "Please provide additional details:",
                placeholder="Add more specific details, dates, sources, or context...",
                height=100,
                key="additional_info_input"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📤 Submit Additional Info", type="primary"):
                    if additional_info.strip():
                        # Following the main.py pattern: append to messages and continue the loop
                        st.session_state.messages.append(additional_info)
                        st.session_state.processing = True
                        st.session_state.workflow_state = 'processing'
                        # Don't clear workflow history, continue from where we left off
                        st.rerun()
            
            with col_b:
                if st.button("🔙 Start Over"):
                    st.session_state.workflow_state = 'input'
                    st.session_state.messages = []
                    st.session_state.workflow_history = []
                    st.session_state.clarification_message = ""
                    st.rerun()
        
        # Completed section
        elif st.session_state.workflow_state == 'completed':
            st.subheader("✅ Fact-Check Complete")
            
            if st.session_state.final_report:
                st.success("Your fact-check report has been generated!")
                
                # Display the report
                with st.container():
                    st.markdown("### 📊 Fact-Check Report")
                    st.markdown(st.session_state.final_report)
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=st.session_state.final_report,
                    file_name=f"fact_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            if st.button("🔄 Check Another Claim"):
                st.session_state.workflow_state = 'input'
                st.session_state.messages = []
                st.session_state.final_report = None
                st.session_state.workflow_history = []
                st.rerun()
        
        # Error section
        elif st.session_state.workflow_state == 'error':
            st.subheader("❌ Error Occurred")
            st.error("There was an issue processing your fact-check request.")
            
            if st.button("🔄 Try Again"):
                st.session_state.workflow_state = 'input'
                st.session_state.messages = []
                st.session_state.workflow_history = []
                st.rerun()
    
    with col2:
        # Progress and history display
        if st.session_state.workflow_history:
            display_workflow_history()
        
        # Show claim statement if available
        if st.session_state.claim_statement:
            st.subheader("📝 Extracted Claim")
            st.info(st.session_state.claim_statement)

if __name__ == "__main__":
    main()