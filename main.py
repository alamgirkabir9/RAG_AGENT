import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.agent import RAGAgent

def main():
    """Main entry point for the RAG Agent"""
    print("🤖 RAG Agent Starting...")
    print("Type 'quit' to exit\n")
    
    agent = RAGAgent()
    
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            response = agent.process_query(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()