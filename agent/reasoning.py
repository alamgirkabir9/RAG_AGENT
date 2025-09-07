import re
from typing import Dict, Any

class Reasoner:
    """Decides which tool to use based on user query"""
    
    def decide_tool(self, query: str, context: str = "") -> Dict[str, Any]:
        """Analyze query and decide which tool to use"""
        
        query_lower = query.lower()
        
        # Check for mathematical operations (including written numbers)
        if self._is_math_query(query_lower):
            return {"tool": "calculator", "confidence": 0.9}
        
        # Check for web search indicators
        if self._is_web_search_query(query_lower):
            return {"tool": "web_search", "confidence": 0.8}
        
        # Always try RAG first for knowledge queries (if documents exist)
        if self._might_be_in_knowledge_base(query_lower):
            return {"tool": "rag_search", "confidence": 0.7}
        
        # Default to Groq chat for everything else
        return {"tool": "groq_chat", "confidence": 0.6}
    
    def _is_math_query(self, query: str) -> bool:
        """Check if query contains mathematical operations"""
        # Written number words with word boundaries
        number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        math_words = ['plus', 'minus', 'add', 'subtract', 'multiply', 'divide', 'times']
        
        # Check for written math expressions with word boundaries
        has_numbers = any(re.search(r'\b' + word + r'\b', query) for word in number_words) or bool(re.search(r'\d+', query))
        has_math_words = any(re.search(r'\b' + word + r'\b', query) for word in math_words)
        
        if has_numbers and has_math_words:
            return True
            
        math_patterns = [
            r'\d+\s*[\+\-\*/\^]\s*\d+',  # Basic arithmetic with symbols
            r'\b(calc?ulate?|compute|solve|math|equation)\b',  # Handle typos like "calc", "clac", "calcul"
            r'what is \d+.*\d+',
            r'square root|sqrt|factorial|log|sin|cos|tan',
            r'\d+\s+(divided|div)\s+\d+',  # Handle "6 divided 3" without "by"
            r'\d+\s+divided\s+by\s+\d+',  # Standard "divided by"
            r'\d+\s*[/÷]\s*\d+',  # Division symbols
            r'\d+\s+(plus|add|added|sum)\s+\d+',  # Addition variations
            r'\d+\s+(minus|subtract|sub)\s+\d+',  # Subtraction variations
            r'\d+\s+(times|multiply|mul|x)\s+\d+'  # Multiplication variations
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in math_patterns)
    
    def _is_web_search_query(self, query: str) -> bool:
        """Check if query requires web search"""
        query = query.lower()

        web_indicators = [
            r'\blatest\b',
            r'\brecent\b',
            r'\bcurrent\b',
            r'\btoday\b',
            r'\bnews?\b',  # matches 'news' and 'new'
            r'what happened',
            r'weather',
            r'stock price',
            r'search for',
            r'find information (on|about)',
            r"what('s| is) new",
            r'update (on|about)',
            r'recent developments?',
            r'\bheadlines?\b',
            r'\bevents?\b'
        ]

        return any(re.search(pattern, query) for pattern in web_indicators)

    def _might_be_in_knowledge_base(self, query: str) -> bool:
        """Check if query might be answerable by knowledge base"""
        # Skip very short queries or greetings
        if len(query.split()) < 2:
            return False
            
        # Skip obvious greetings/social
        social_patterns = [
            'hello', 'hi ', 'hey', 'good morning', 'good afternoon', 'good evening',
            'thank you', 'thanks', 'bye', 'goodbye', 'how are you'
        ]
        
        if any(pattern in query for pattern in social_patterns):
            return False
        
        # Check if query is about topics likely in AI/tech knowledge base
        ai_tech_terms = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'computer vision', 'nlp', 'natural language',
            'algorithm', 'robotics', 'expert system', 'automation',
            'data science', 'python', 'programming', 'software', 'technology'
        ]
        
        query_lower = query.lower()
        if any(term in query_lower for term in ai_tech_terms):
            return True
        
        # Check for country/geography queries that should go to web search
        geography_terms = [
            'china', 'usa', 'country', 'nation', 'city', 'capital',
            'population', 'economy', 'politics', 'government', 'history of',
            'culture', 'language', 'religion', 'geography'
        ]
        
        if any(term in query_lower for term in geography_terms):
            return False  # Send to web search instead
        
        # For other general queries, try RAG first
        return True