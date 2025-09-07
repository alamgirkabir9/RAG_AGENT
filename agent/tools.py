from typing import Dict, Any
import sys
import os

# Add tools directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))

from tools.calc_tool import Calculator
from tools.rag_tool import RAGTool
from tools.search_tool import WebSearchTool
from tools.groq_api import run_groq

class ToolManager:
    """Manages and executes different tools"""
    
    def __init__(self):
        self.calculator = Calculator()
        self.rag_tool = RAGTool()
        self.web_search = WebSearchTool()
    
    def execute_tool(self, tool_name: str, query: str, params: Dict[str, Any] = {}) -> str:
        """Execute the specified tool with the given query"""
        
        try:
            if tool_name == "calculator":
                print("From Calculator Tool: ")
                return self.calculator.calculate(query)
            
            elif tool_name == "rag_search":
                result = self.rag_tool.search_and_answer(query)
                
                # Check if RAG suggests fallback to Groq
                if isinstance(result, dict) and result.get("fallback", False):
                    print(f"🔄 RAG fallback: {result['message']}")
                    print("🤖 Using Groq chat instead...")
                    return run_groq(query)
                elif isinstance(result, dict):
                    return result["message"]
                else:
                    return result
            
            elif tool_name == "web_search":
                return self.web_search.search(query)
            
            elif tool_name == "groq_chat":
                return run_groq(query)
            
            else:
                return f"Unknown tool: {tool_name}. Using Groq fallback..."
                
        except Exception as e:
            print(f"⚠️ Error with {tool_name}: {str(e)}")
            print("🤖 Falling back to Groq chat...")
            return run_groq(query)
