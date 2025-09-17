# RAG Agent - Enhanced Conversational AI with Memory

An intelligent conversational agent that combines Retrieval-Augmented Generation (RAG) with advanced memory management, contextual awareness, and multi-tool integration.

## 🚀 Features

### Core Capabilities
- **Advanced Memory Management**: Persistent long-term and session-based short-term memory
- **Multi-Tool Integration**: Calculator, RAG search, web search, and Groq chat
- **Contextual Intelligence**: Maintains conversation continuity with pronoun resolution
- **Personal Information Tracking**: Learns and remembers user preferences and facts
- **Adaptive Processing**: Adjusts behavior based on query type and conversation history

### Memory System
- **Short-term Memory**: Recent conversation context (configurable, default 15 messages)
- **Long-term Memory**: Persistent facts, preferences, and user information (default 200 entries)
- **Session Tracking**: Conversation statistics and interaction patterns
- **Auto-backup**: Automatic memory export every 50 messages

### Processing Modes
- **Default**: Standard conversational responses
- **Personal**: Personalized responses using stored user information
- **Research**: Detailed, well-structured information responses
- **Creative**: Imaginative responses incorporating user context

## 🏗️ Architecture

```
RAGAgent
├── Memory Management (memory.py)
│   ├── Short-term conversation buffer
│   ├── Long-term fact storage
│   └── Session persistence
├── Reasoning Engine (reasoning.py)
│   ├── Query analysis
│   ├── Tool selection logic
│   └── Context dependency detection
├── Tool Manager (tools.py)
│   ├── Calculator
│   ├── RAG Search
│   ├── Web Search
│   └── Groq Chat
└── Enhanced Processing Pipeline
    ├── Information extraction
    ├── Context building
    ├── Response post-processing
    └── Pattern learning
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Required dependencies (install via pip)

### Setup
```bash
# Clone or download the project
git clone <repository-url>
cd rag-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_key
# Other service API keys as needed

# Run the agent
python main.py
```

### Required Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key_here
# Add other API keys for web search, RAG services, etc.
```

## 🛠️ Configuration

The agent accepts a configuration dictionary on initialization:

```python
config = {
    "max_short_term": 15,        # Short-term memory size
    "max_long_term": 200,        # Long-term memory size
    "memory_file": "agent_memory.json",    # Memory persistence file
    "session_file": "agent_session.json", # Session data file
    "context_window": 20,        # Context size for processing
    "groq_enhanced": True,       # Enable Groq enhancements
    "auto_export": True,         # Auto-backup memory
    "debug_mode": False          # Enable debug output
}

agent = RAGAgent(config=config)
```

## 💬 Usage Examples

### Basic Conversation
```
You: What's the weather like today?
Agent: I'll search for current weather information for you.
```

### Personal Information Learning
```
You: My name is John and I work as a software engineer
Agent: Nice to meet you, John! I'll remember that you're a software engineer.

You: What do you know about me?
Agent: You're John, and you work as a software engineer. [Additional stored information]
```

### Context-Aware Queries
```
You: Tell me about France
Agent: France is a country in Western Europe... [detailed information]

You: What's its capital?
Agent: The capital of France is Paris. [Correctly resolves "its" to France]
```

### Mathematical Calculations
```
You: Calculate 15% of 250
Agent: 15% of 250 = 37.5
```

## 🎛️ Special Commands

The agent supports several special commands for memory and system management:

- `memory stats` - Display comprehensive memory statistics
- `clear memory` - Clear all stored memory (use with caution)
- `clear session` - Clear current session, keep long-term facts
- `search memory <query>` - Search through stored memory
- `export memory` - Export memory to backup file
- `show tools` - Display available tools
- `user profile` - Show your stored profile information
- `conversation summary` - Get a summary of the current session
- `help` or `commands` - Show available commands

## 🧠 How It Works

### 1. Query Processing Pipeline
1. **Special Command Detection**: Check for system commands first
2. **Query Analysis**: Analyze query type, complexity, and characteristics
3. **Mode Adjustment**: Set processing mode based on analysis
4. **Information Extraction**: Extract personal info using Groq (if present)
5. **Memory Integration**: Add query to memory with metadata
6. **Context Building**: Gather relevant conversation and stored context
7. **Tool Selection**: Choose appropriate tool with enhanced logic
8. **Execution**: Run selected tool with full context
9. **Post-processing**: Enhance and format response
10. **Memory Update**: Store response and update user patterns

### 2. Tool Selection Logic
The agent uses sophisticated logic to choose the right tool:

- **Context-dependent queries** (using "it", "that", etc.) → Groq Chat with full context
- **Mathematical expressions** → Calculator
- **Explicit web searches** → Web Search
- **Document/RAG requests** → RAG Search
- **General knowledge** → RAG Search first, fallback to Groq Chat
- **Personal queries** → Enhanced personal response with stored data

### 3. Memory Management
- **Automatic Extraction**: Uses Groq to extract personal information from conversations
- **Categorization**: Organizes information by type (identity, preferences, professional, etc.)
- **Deduplication**: Prevents storing duplicate information
- **Persistence**: Saves memory to JSON files for session continuity
- **Smart Retrieval**: Finds relevant context based on query similarity

## 📊 Memory Structure

### Long-term Memory Format
```json
{
  "important_facts": [
    {
      "type": "identity",
      "value": "Software Engineer",
      "timestamp": "2024-01-15T10:30:00",
      "confidence": 0.9,
      "source_query": "I work as a software engineer"
    }
  ],
  "user_preferences": {
    "likes": ["programming", "coffee"],
    "dislikes": ["meetings"],
    "favorites": ["Python"]
  },
  "topics_discussed": {
    "programming": 5,
    "weather": 2,
    "france": 1
  }
}
```

### Session Memory Format
```json
[
  {
    "role": "user",
    "content": "What's the capital of France?",
    "timestamp": "2024-01-15T10:30:00",
    "metadata": {
      "query_type": "search",
      "complexity": "simple",
      "session_id": "20240115_103000"
    }
  }
]
```

## 🔧 Debugging

Enable debug mode for detailed processing information:

```python
agent = RAGAgent(config={"debug_mode": True})
```

Debug output includes:
- Query processing steps
- Tool selection reasoning
- Context retrieval details
- Memory operations
- Error traces

## 🤝 Contributing

Areas for improvement:
- Add more specialized tools
- Enhance information extraction accuracy
- Improve context window management
- Add conversation export formats
- Implement user authentication
- Add conversation analytics

## 📝 License

[Add your license information here]

## 🐛 Troubleshooting

### Common Issues

**Memory not persisting between sessions**
- Check file permissions for memory files
- Verify the agent has write access to the directory

**Tool selection seems incorrect**
- Enable debug mode to see decision reasoning
- Check if context is being properly retrieved

**Groq API errors**
- Verify your GROQ_API_KEY in .env file
- Check API rate limits and quotas

**Performance issues with large memory**
- Consider reducing max_long_term setting
- Implement memory cleanup routines

## 🔮 Future Enhancements

- Vector-based memory retrieval for better context matching
- Multi-modal support (images, documents)
- Conversation branching and threading
- Advanced user modeling and personality adaptation
- Integration with external knowledge bases
- Real-time learning and fact verification
