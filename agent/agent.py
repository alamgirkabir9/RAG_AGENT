import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from .memory import Memory
from .reasoning import Reasoner
from .tools import ToolManager
import json
import re
from datetime import datetime

load_dotenv()

class RAGAgent:
    """Main agent that coordinates all tools and memory with Groq-powered pattern recognition"""
    
    def __init__(self):
        # Initialize components
        self.memory = Memory(
            max_short_term=15,
            max_long_term=200,
            memory_file="agent_memory.json"
        )
        self.reasoner = Reasoner()
        self.tool_manager = ToolManager()
        
        print("🧠 Agent initialized with Groq-powered memory")
        print("🛠️ Tools loaded: Calculator, RAG Search, Web Search, Groq Chat")
        
        # Load and display existing memory if any
        stats = self.memory.get_memory_stats()
        if stats['long_term_facts'] > 0:
            print(f"📚 Loaded {stats['long_term_facts']} facts from previous sessions")
    
    def process_query(self, user_input: str) -> str:
        """Process user query with Groq-powered memory analysis"""
        
        # Step 1: Use Groq to analyze the message for information extraction
        extracted_info = self._extract_information_with_groq(user_input)
        if extracted_info:
            self._save_extracted_info_to_memory(extracted_info, user_input)
        
        # Step 2: Add user message to memory (this will also trigger built-in extraction)
        self.memory.add_message("user", user_input)
        
        # Step 3: Check if this is a personal query using Groq
        if self._is_personal_query_groq(user_input):
            personal_response = self._handle_personal_query_groq(user_input)
            if personal_response:
                self.memory.add_message("assistant", personal_response)
                return personal_response
        
        # Step 4: Decide which tool to use (if not handled as personal query)
        decision = self.reasoner.decide_tool(user_input)
        tool_name = decision["tool"]
        
        # Get context for the query
        full_context = self.memory.get_full_context(user_input)
        
        try:
            # For groq_chat, enhance with memory context
            if tool_name == "groq_chat":
                enhanced_query = self._enhance_query_with_context(user_input, full_context)
                response = self.tool_manager.execute_tool(tool_name, enhanced_query)
            else:
                response = self.tool_manager.execute_tool(tool_name, user_input)
            
            # Add response to memory
            self.memory.add_message("assistant", response)
            return response
            
        except Exception as e:
            error_response = f"❌ Error processing query: {str(e)}"
            self.memory.add_message("assistant", error_response)
            return error_response
    
    def _extract_information_with_groq(self, user_input: str) -> dict:
        """Use Groq to extract personal information from user input"""
        
        extraction_prompt = f"""Analyze the following user message and extract any personal information that should be remembered for future conversations.

User message: "{user_input}"

Please identify and extract:
1. Name (if the user mentions their name)
2. Job/Work (if they mention their job, company, role)
3. Location (if they mention where they live/work)
4. Preferences (things they like, dislike, prefer)
5. Personal facts (age, hobbies, family, etc.)

Respond ONLY with a JSON object in this exact format:
{{
    "has_personal_info": true/false,
    "extracted_data": {{
        "name": "extracted name or null",
        "job": "extracted job info or null", 
        "location": "extracted location or null",
        "preferences": ["list of things they mentioned liking/preferring"],
        "facts": ["list of other personal facts mentioned"]
    }}
}}

If no personal information is found, return {{"has_personal_info": false, "extracted_data": {{}}}}"""

        try:
            response = self.tool_manager.execute_tool("groq_chat", extraction_prompt)
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                if extracted.get("has_personal_info"):
                    print(f"🧠 Groq extracted info: {extracted['extracted_data']}")
                    return extracted
            
        except Exception as e:
            print(f"Debug: Info extraction failed: {e}")
        
        return None
    
    def _save_extracted_info_to_memory(self, extracted_info: dict, original_input: str):
        """Save extracted information using Memory's built-in extraction by creating targeted messages"""
        data = extracted_info.get("extracted_data", {})
        
        # Create messages that will trigger Memory's built-in pattern recognition
        messages_to_process = []
        
        if data.get("name"):
            messages_to_process.append(f"My name is {data['name']}")
            
        if data.get("job"):
            messages_to_process.append(f"I work {data['job']}")
            
        if data.get("location"):
            messages_to_process.append(f"I live in {data['location']}")
            
        if data.get("preferences"):
            for pref in data["preferences"]:
                messages_to_process.append(f"I like {pref}")
        
        # Process each message through Memory's extraction system
        for msg_content in messages_to_process:
            fake_message = {
                "role": "user",
                "content": msg_content,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"extracted_by_groq": True}
            }
            
            # Use Memory's built-in extraction methods
            if self.memory._should_preserve_long_term(fake_message):
                self.memory._move_to_long_term(fake_message)
            
        if messages_to_process:
            self.memory._save_long_term_memory()
            print(f"💾 Groq enhanced memory with: {', '.join(messages_to_process)}")
    
    def _is_personal_query_groq(self, user_input: str) -> bool:
        """Use Groq to determine if this is a personal information query"""
        
        detection_prompt = f"""Analyze this user message and determine if they are asking for personal information about themselves that should be retrieved from memory.

User message: "{user_input}"

Examples of personal queries:
- "What is my name?"
- "Who am I?"
- "What do I like?"
- "Where do I work?"
- "Tell me about myself"
- "What do you know about me?"
- "Write a poem using my name"
- "Make something personal for me"
- "What are my preferences?"

Respond with ONLY: YES or NO"""

        try:
            response = self.tool_manager.execute_tool("groq_chat", detection_prompt)
            is_personal = "YES" in response.upper()
            if is_personal:
                print(f"🔍 Groq detected personal query: {user_input}")
            return is_personal
        except:
            return False
    
    def _handle_personal_query_groq(self, user_input: str) -> str:
        """Use Groq to handle personal queries with memory context"""
        
        # Get user summary from memory
        summary = self.memory.get_user_summary()
        
        # Build memory context string
        memory_context = self._build_memory_context(summary)
        
        if not memory_context.strip():
            return "I don't know much about you yet! Please tell me about yourself - your name, what you like, where you work, etc. 😊"
        
        personal_prompt = f"""You are a helpful AI assistant answering a personal question from a user. Use the information I have stored about them to provide a relevant, personalized response.

=== STORED INFORMATION ABOUT THE USER ===
{memory_context}

=== USER'S QUESTION ===
{user_input}

Instructions:
- Use the stored information to answer their question
- Be friendly and personal
- If asking for a poem or creative content using their name, create it using their stored name
- If asking what you know about them, provide a nice summary
- If the stored information doesn't contain what they're asking for, let them know what you do/don't know
- Keep responses warm and conversational

Respond as their helpful AI assistant:"""

        try:
            response = self.tool_manager.execute_tool("groq_chat", personal_prompt)
            print(f"🤖 Generated personal response using stored memory")
            return response
        except Exception as e:
            return f"❌ Error accessing your personal information: {str(e)}"
    
    def _build_memory_context(self, summary: dict) -> str:
        """Build a formatted memory context string"""
        context_parts = []
        
        # Add facts from important_facts
        important_facts = self.memory.long_term_memory.get("important_facts", [])
        if important_facts:
            context_parts.append("Personal Facts:")
            for fact in important_facts[:10]:  # Show up to 10 facts
                context_parts.append(f"  - {fact['type']}: {fact['value']}")
            context_parts.append("")
        
        # Add preferences from user_preferences
        user_preferences = self.memory.long_term_memory.get("user_preferences", {})
        if user_preferences:
            context_parts.append("User Preferences:")
            for category, items in user_preferences.items():
                if isinstance(items, list) and items:
                    context_parts.append(f"  - {category}: {', '.join(items[:5])}")
                elif isinstance(items, str):
                    context_parts.append(f"  - {category}: {items}")
            context_parts.append("")
        
        # Add conversation stats
        if summary.get("conversation_count", 0) > 0:
            context_parts.append(f"We've had {summary['conversation_count']} messages in this session.")
        
        # Add top topics
        if summary.get("top_topics"):
            top_topics = summary["top_topics"][:3]
            topics_text = ", ".join([f"{topic} ({count}x)" for topic, count in top_topics])
            context_parts.append(f"Most discussed topics: {topics_text}")
        
        return "\n".join(context_parts)
    
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
  
🤖 Groq-Powered Features:
  🧠 Automatic information extraction from conversations
  🔍 Smart personal query detection
  💭 Context-aware personal responses using your Memory class
        """
        print(tools_info)
