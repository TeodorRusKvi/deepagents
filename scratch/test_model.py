from dotenv import load_dotenv
import os
load_dotenv("/Users/teodorrustadkvisberg/projects/current/deepagents/.env")

# Test create_deep_agent
print("\nTesting create_deep_agent...")
try:
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import MemorySaver
    from pathlib import Path
    
    workspace_root = Path("/Users/teodorrustadkvisberg/projects/current/deepagents")
    model = "openai:gpt-4o-mini"
    
    agent = create_deep_agent(
        model=model,
        system_prompt="Test prompt",
        checkpointer=MemorySaver(),
    )
    print("✅ Successfully created deep agent")
    
    # Try a simple invocation
    print("Attempting agent invocation...")
    for chunk in agent.stream({"messages": [("user", "hi")]}, config={"configurable": {"thread_id": "test"}}):
        print(f"Chunk: {chunk}")
    print("✅ Agent invocation successful")
    
except Exception as e:
    import traceback
    print(f"❌ Failed to test deep agent: {e}")
    traceback.print_exc()
