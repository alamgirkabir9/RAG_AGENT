from .memory import Memory
from .reasoning import Reasoner
from .tools import ToolManager

class RAGAgent:
    """Main RAG Agent that orchestrates the workflow"""
    
    def __init__(self):
        self.memory = Memory()
        self.reasoner = Reasoner()
        self.tool_manager = ToolManager()
        
    def process_query(self, query: str) -> str:
        """Process user query through the agent workflow"""
        
        # Store user query in memory
        self.memory.add_message("user", query)
        
        # Determine which tool to use
        tool_decision = self.reasoner.decide_tool(query, self.memory.get_recent_context())
        
        # Execute the appropriate tool
        result = self.tool_manager.execute_tool(tool_decision['tool'], query, tool_decision.get('params', {}))
        
        # Store agent response in memory
        self.memory.add_message("agent", result)
        
        return result
