# agent/agent.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from .memory import Memory
from .reasoning import Reasoner
from .tools import ToolManager

load_dotenv()

class RAGAgent:
    """Main agent that coordinates all tools and memory"""
    
    def __init__(self):
        # Initialize components
        self.memory = Memory(
            max_short_term=15,
            max_long_term=200,
            memory_file="agent_memory.json"
        )
        self.reasoner = Reasoner()
        self.tool_manager = ToolManager()
        
        print("🧠 Agent initialized with persistent memory")
        print("🛠️ Tools loaded: Calculator, RAG Search, Web Search, Groq Chat")
        
        # Load and display existing memory if any
        stats = self.memory.get_memory_stats()
        if stats['long_term_facts'] > 0:
            print(f"📚 Loaded {stats['long_term_facts']} facts from previous sessions")
    
    def process_query(self, user_input: str) -> str:
        """Process user query with memory context"""
        
        # Add user message to memory
        self.memory.add_message("user", user_input)
        
        # Check if user is asking about themselves first
        if self._is_personal_query(user_input):
            personal_response = self._handle_personal_query(user_input)
            if personal_response:
                # Add agent response to memory
                self.memory.add_message("assistant", personal_response)
                return personal_response
        
        # Decide which tool to use
        decision = self.reasoner.decide_tool(user_input)
        tool_name = decision["tool"]
        
        # Get context for the query (for groq_chat tool)
        full_context = self.memory.get_full_context(user_input)
        
        try:
            # For groq_chat, we need to enhance the query with memory context
            if tool_name == "groq_chat":
                enhanced_query = self._enhance_query_with_context(user_input, full_context)
                response = self.tool_manager.execute_tool(tool_name, enhanced_query)
            else:
                # For other tools, use the original query
                response = self.tool_manager.execute_tool(tool_name, user_input)
            
            # Add response to memory
            self.memory.add_message("assistant", response)
            return response
            
        except Exception as e:
            error_response = f"❌ Error processing query: {str(e)}"
            self.memory.add_message("assistant", error_response)
            return error_response
    
    def _is_personal_query(self, query: str) -> bool:
        """Check if query is asking about personal information"""
        query_lower = query.lower()
        
        personal_patterns = [
            r'what is my name',
            r'who am i',
            r'my name is',
            r'what do i like',
            r'what do i prefer',
            r'where do i work',
            r'where do i live',
            r'what is my job',
            r'what is my role',
            r'remind me',
            r'do you remember',
            r'what did i tell you about',
            r'what do you know about me'
        ]
        
        import re
        return any(re.search(pattern, query_lower) for pattern in personal_patterns)
    
    def _handle_personal_query(self, query: str) -> str:
        """Handle queries about personal information"""
        query_lower = query.lower()
        
        # Check what user is asking for
        if 'my name' in query_lower or 'who am i' in query_lower:
            # Look for name in long-term memory
            for fact in self.memory.long_term_memory["important_facts"]:
                if fact["type"] == "name":
                    return f"Your name is {fact['value']}! 😊"
            
            return "I don't know your name yet. Could you tell me your name?"
        
        elif 'my job' in query_lower or 'where do i work' in query_lower:
            for fact in self.memory.long_term_memory["important_facts"]:
                if fact["type"] in ["job", "work", "role", "company"]:
                    return f"You work as {fact['value']} 💼"
            
            return "I don't have information about your job. Could you tell me about your work?"
        
        elif 'where do i live' in query_lower:
            for fact in self.memory.long_term_memory["important_facts"]:
                if fact["type"] == "location":
                    return f"You live in {fact['value']} 🏠"
            
            return "I don't know where you live. Could you tell me your location?"
        
        elif 'what do i like' in query_lower or 'what do i prefer' in query_lower:
            preferences = self.memory.long_term_memory["user_preferences"]
            if preferences:
                response_parts = []
                if "likes" in preferences:
                    response_parts.append(f"You like: {', '.join(preferences['likes'][:3])}")
                if "favorites" in preferences:
                    for key, value in preferences.items():
                        if key.startswith("favorite_"):
                            category = key.replace("favorite_", "")
                            response_parts.append(f"Your favorite {category} is {value}")
                
                if response_parts:
                    return "Here's what I know about your preferences:\n" + "\n".join(response_parts) + " 😊"
            
            return "I don't have information about your preferences yet. Tell me what you like!"
        
        elif 'what do you know about me' in query_lower:
            return self._generate_user_summary()
        
        return None  # Let other tools handle it
    
    def _generate_user_summary(self) -> str:
        """Generate a summary of what the agent knows about the user"""
        summary = self.memory.get_user_summary()
        
        if not summary["facts"] and not summary["preferences"]:
            return "I don't know much about you yet! Tell me about yourself - your name, what you like, where you work, etc. 😊"
        
        response_parts = ["Here's what I know about you:\n"]
        
        # Add facts
        if summary["facts"]:
            response_parts.append("📋 Personal Information:")
            for fact in summary["facts"][:5]:  # Show top 5 facts
                response_parts.append(f"  • {fact}")
            response_parts.append("")
        
        # Add preferences
        if summary["preferences"]:
            response_parts.append("❤️ Your Preferences:")
            for category, items in summary["preferences"].items():
                if isinstance(items, list) and items:
                    response_parts.append(f"  • {category}: {', '.join(items[:3])}")
                elif isinstance(items, str):
                    response_parts.append(f"  • {category}: {items}")
            response_parts.append("")
        
        # Add conversation stats
        response_parts.append(f"💬 We've had {summary['conversation_count']} messages in this session")
        
        if summary["top_topics"]:
            top_3_topics = summary["top_topics"][:3]
            topics_text = ", ".join([f"{topic} ({count}x)" for topic, count in top_3_topics])
            response_parts.append(f"🔤 Most discussed: {topics_text}")
        
        return "\n".join(response_parts)
    
    def _enhance_query_with_context(self, query: str, context: str) -> str:
        """Enhance query with memory context for better responses"""
        
        if not context:
            return query
        
        # Build enhanced prompt
        enhanced_query = f"""You are a helpful AI assistant with access to our conversation history and information about the user.
        
=== CONTEXT FROM OUR CONVERSATION ===
{context}

=== CURRENT QUESTION ===
{query}

Please provide a helpful response that takes into account what you know about the user from our conversation history. If you know personal details about them (name, preferences, job, etc.), reference them naturally when relevant."""

        return enhanced_query
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics"""
        return self.memory.get_memory_stats()
    
    def clear_memory(self):
        """Clear all memory"""
        self.memory.clear_all_memory()
        print("🧠 Memory cleared!")
    
    def export_memory(self, filename: str = "memory_backup.json"):
        """Export memory to file"""
        self.memory.export_memory(filename)
    
    def search_memory(self, query: str):
        """Search through memory"""
        results = self.memory.search_memory(query)
        if results:
            print(f"\n🔍 Found {len(results)} memory matches:")
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. [{result['type']}] {result['content'][:100]}...")
        else:
            print("🔍 No matches found in memory")
    
    def show_available_tools(self):
        """Show available tools"""
        tools_info = """
🛠️ Available Tools:
  📊 Calculator - Mathematical calculations 
  📚 RAG Search - Search through documents/knowledge base
  🔍 Web Search - Search the internet for current information
  🤖 Groq Chat - General conversation with AI (with memory context)
  🧠 Memory Commands - 'memory stats', 'clear memory', 'search <query>'
        """
        print(tools_info)
