from typing import List, Callable
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

def build_lcel_agent_executor(llm, tools: List[BaseTool], prompt: ChatPromptTemplate, callbacks: List[Callable] = None):
    """
    Builds a LangChain Expression Language (LCEL) AgentExecutor.
    Replaces the legacy while-loop ToolCallingAgent.
    """
    if callbacks is None:
        callbacks = []
        
    # Create the LCEL tool calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Wrap in AgentExecutor which handles the actual while loop execution natively in LangChain
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=5,
        return_intermediate_steps=True,
        callbacks=callbacks
    )
    
    return executor
