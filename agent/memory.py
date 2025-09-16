from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
import re
from collections import defaultdict

class Memory:
    """Advanced memory management with short-term and long-term storage"""
    
    def __init__(self, 
                 max_short_term: int = 10,
                 max_long_term: int = 100,
                 memory_file: str = "memory.json"):
        
        # Short-term memory: Recent conversation context
        self.short_term_memory: List[Dict] = []
        self.max_short_term = max_short_term
        
        # Long-term memory: Important facts, preferences, and key information
        self.long_term_memory: Dict = {
            "user_preferences": {},
            "important_facts": [],
            "topics_discussed": defaultdict(int),
            "user_context": {},
            "key_conversations": []
        }
        
        self.max_long_term = max_long_term
        self.memory_file = memory_file
        
        # Load existing long-term memory
        self._load_long_term_memory()
        
        # Keywords that indicate important information
        self.importance_keywords = [
            "my name is", "i am", "i work", "i live", "i like", "i hate", 
            "i prefer", "remember", "important", "always", "never",
            "my job", "my role", "my company", "my project"
        ]
        
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to short-term memory and analyze for long-term storage"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to short-term memory
        self.short_term_memory.append(message)
        
        # Maintain short-term memory size
        if len(self.short_term_memory) > self.max_short_term:
            # Before removing, check if the oldest message should be preserved
            oldest = self.short_term_memory[0]
            if self._should_preserve_long_term(oldest):
                self._move_to_long_term(oldest)
            
            self.short_term_memory = self.short_term_memory[-self.max_short_term:]
        
        # Analyze current message for long-term storage
        if role == "user" and self._should_preserve_long_term(message):
            self._move_to_long_term(message)
        
        # Update topic tracking
        self._update_topic_tracking(content)
        
        # Save long-term memory periodically
        self._save_long_term_memory()
    
    def _should_preserve_long_term(self, message: Dict) -> bool:
        """Determine if a message should be stored in long-term memory"""
        content = message["content"].lower()
        
        # Check for importance keywords
        if any(keyword in content for keyword in self.importance_keywords):
            return True
        
        # Check for questions that might need context later
        if any(pattern in content for pattern in [
            "how do i", "what is my", "remind me", "my preference",
            "i told you", "as i mentioned"
        ]):
            return True
        
        # Check for definitive statements about user
        if re.search(r'\b(i am|i\'m|my|mine)\b', content) and len(content.split()) > 5:
            return True
            
        return False
    
    def _move_to_long_term(self, message: Dict):
        """Move important information to long-term memory"""
        content = message["content"].lower()
        
        # Extract user preferences
        self._extract_preferences(message)
        
        # Extract facts about the user
        self._extract_user_facts(message)
        
        # Store important conversations
        if len(self.long_term_memory["key_conversations"]) < 20:
            self.long_term_memory["key_conversations"].append({
                "content": message["content"],
                "timestamp": message["timestamp"],
                "importance_score": self._calculate_importance(message)
            })
    
    def _extract_preferences(self, message: Dict):
        """Extract user preferences from message"""
        content = message["content"].lower()
        
        preference_patterns = [
            (r'i (like|love|enjoy|prefer) (.+)', 'likes'),
            (r'i (hate|dislike|don\'t like) (.+)', 'dislikes'),
            (r'i always (.+)', 'habits'),
            (r'i never (.+)', 'never_do'),
            (r'my favorite (.+) is (.+)', 'favorites')
        ]
        
        for pattern, category in preference_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if category == 'favorites':
                    self.long_term_memory["user_preferences"][f"favorite_{match[0]}"] = match[1]
                else:
                    if category not in self.long_term_memory["user_preferences"]:
                        self.long_term_memory["user_preferences"][category] = []
                    
                    preference_text = match if isinstance(match, str) else match[-1]
                    if preference_text not in self.long_term_memory["user_preferences"][category]:
                        self.long_term_memory["user_preferences"][category].append(preference_text)
    
    def _extract_user_facts(self, message: Dict):
        """Extract factual information about the user"""
        content = message["content"].lower()
        
        fact_patterns = [
            (r'my name is (.+)', 'name'),
            (r'i am (.+)', 'identity'),
            (r'i work (.+)', 'work'),
            (r'i live in (.+)', 'location'),
            (r'my job is (.+)', 'job'),
            (r'my role is (.+)', 'role'),
            (r'my company is (.+)', 'company')
        ]
        
        for pattern, fact_type in fact_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                fact = {
                    "type": fact_type,
                    "value": match.strip(),
                    "timestamp": message["timestamp"],
                    "confidence": 0.8
                }
                
                # Avoid duplicates
                existing = [f for f in self.long_term_memory["important_facts"] 
                           if f["type"] == fact_type]
                if not existing:
                    self.long_term_memory["important_facts"].append(fact)
                else:
                    # Update if more recent
                    existing[0]["value"] = match.strip()
                    existing[0]["timestamp"] = message["timestamp"]
    
    def _update_topic_tracking(self, content: str):
        """Track topics being discussed"""
        # Simple topic extraction (you could use more sophisticated NLP here)
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 
            'how', 'why', 'when', 'where', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'this', 'that', 'these', 'those'
        }
        
        topics = [word for word in words 
                 if len(word) > 3 and word not in stop_words]
        
        for topic in topics:
            self.long_term_memory["topics_discussed"][topic] += 1
    
    def _calculate_importance(self, message: Dict) -> float:
        """Calculate importance score for a message"""
        content = message["content"].lower()
        score = 0.0
        
        # Length bonus (longer messages often more important)
        score += min(len(content.split()) / 50, 0.3)
        
        # Keyword bonus
        for keyword in self.importance_keywords:
            if keyword in content:
                score += 0.2
        
        # Question bonus
        if '?' in content:
            score += 0.1
        
        # Personal information bonus
        if any(word in content for word in ['my', 'i am', 'i work', 'i live']):
            score += 0.3
        
        return min(score, 1.0)
    
    def get_recent_context(self, num_messages: int = 5) -> str:
        """Get recent conversation context (backward compatibility)"""
        return self.get_short_term_context(num_messages)
    
    def get_short_term_context(self, num_messages: int = 5) -> str:
        """Get recent conversation context"""
        recent_messages = self.short_term_memory[-num_messages:]
        
        context = []
        for msg in recent_messages:
            context.append(f"{msg['role'].title()}: {msg['content']}")
        
        return "\n".join(context)
    
    def get_relevant_long_term_context(self, current_query: str) -> str:
        """Get relevant long-term memory based on current query"""
        context_parts = []
        query_lower = current_query.lower()
        
        # Add user preferences if relevant
        if any(word in query_lower for word in ['like', 'prefer', 'favorite', 'hate', 'dislike']):
            if self.long_term_memory["user_preferences"]:
                context_parts.append("User Preferences:")
                for category, items in self.long_term_memory["user_preferences"].items():
                    if isinstance(items, list) and items:
                        context_parts.append(f"- {category}: {', '.join(items[:3])}")
                    elif isinstance(items, str):
                        context_parts.append(f"- {category}: {items}")
        
        # Add relevant user facts
        relevant_facts = []
        for fact in self.long_term_memory["important_facts"]:
            if any(word in query_lower for word in [fact["type"], fact["value"].lower()]):
                relevant_facts.append(f"{fact['type']}: {fact['value']}")
        
        if relevant_facts:
            context_parts.append("User Information:")
            context_parts.extend(f"- {fact}" for fact in relevant_facts[:3])
        
        # Add frequently discussed topics
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        relevant_topics = []
        
        for topic, count in self.long_term_memory["topics_discussed"].items():
            if topic in query_words and count > 2:  # Topic discussed multiple times
                relevant_topics.append((topic, count))
        
        if relevant_topics:
            relevant_topics.sort(key=lambda x: x[1], reverse=True)
            context_parts.append("Previously Discussed:")
            context_parts.extend(f"- {topic} (mentioned {count} times)" 
                               for topic, count in relevant_topics[:3])
        
        return "\n".join(context_parts) if context_parts else ""
    
    def get_full_context(self, current_query: str = "", num_short_term: int = 5) -> str:
        """Get combined short-term and relevant long-term context"""
        contexts = []
        
        # Add relevant long-term context
        long_term = self.get_relevant_long_term_context(current_query)
        if long_term:
            contexts.append("=== Relevant Background ===")
            contexts.append(long_term)
            contexts.append("")
        
        # Add short-term context
        short_term = self.get_short_term_context(num_short_term)
        if short_term:
            contexts.append("=== Recent Conversation ===")
            contexts.append(short_term)
        
        return "\n".join(contexts)
    
    def get_user_summary(self) -> Dict:
        """Get a summary of what we know about the user"""
        summary = {
            "preferences": dict(self.long_term_memory["user_preferences"]),
            "facts": [f"{fact['type']}: {fact['value']}" 
                     for fact in self.long_term_memory["important_facts"]],
            "top_topics": sorted(self.long_term_memory["topics_discussed"].items(), 
                               key=lambda x: x[1], reverse=True)[:10],
            "conversation_count": len(self.short_term_memory),
            "long_term_facts_count": len(self.long_term_memory["important_facts"])
        }
        return summary
    
    def search_memory(self, query: str) -> List[Dict]:
        """Search through both short-term and long-term memory"""
        results = []
        query_lower = query.lower()
        
        # Search short-term memory
        for msg in self.short_term_memory:
            if query_lower in msg["content"].lower():
                results.append({
                    "type": "short_term",
                    "content": msg["content"],
                    "timestamp": msg["timestamp"],
                    "role": msg["role"]
                })
        
        # Search long-term facts
        for fact in self.long_term_memory["important_facts"]:
            if query_lower in fact["value"].lower() or query_lower in fact["type"]:
                results.append({
                    "type": "long_term_fact",
                    "content": f"{fact['type']}: {fact['value']}",
                    "timestamp": fact["timestamp"]
                })
        
        # Search key conversations
        for conv in self.long_term_memory["key_conversations"]:
            if query_lower in conv["content"].lower():
                results.append({
                    "type": "key_conversation",
                    "content": conv["content"],
                    "timestamp": conv["timestamp"]
                })
        
        # Sort by relevance and recency
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results
    
    def forget_old_memories(self, days_threshold: int = 30):
        """Remove old memories from long-term storage"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        # Remove old facts
        self.long_term_memory["important_facts"] = [
            fact for fact in self.long_term_memory["important_facts"]
            if datetime.fromisoformat(fact["timestamp"]) > cutoff_date
        ]
        
        # Remove old conversations
        self.long_term_memory["key_conversations"] = [
            conv for conv in self.long_term_memory["key_conversations"]
            if datetime.fromisoformat(conv["timestamp"]) > cutoff_date
        ]
        
        self._save_long_term_memory()
    
    def _load_long_term_memory(self):
        """Load long-term memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    saved_memory = json.load(f)
                    self.long_term_memory.update(saved_memory)
                    # Convert defaultdict back
                    self.long_term_memory["topics_discussed"] = defaultdict(
                        int, self.long_term_memory.get("topics_discussed", {})
                    )
                print(f"📚 Loaded long-term memory with {len(self.long_term_memory['important_facts'])} facts")
            except Exception as e:
                print(f"⚠️ Error loading memory: {e}")
    
    def _save_long_term_memory(self):
        """Save long-term memory to file"""
        try:
            # Convert defaultdict to regular dict for JSON serialization
            memory_to_save = self.long_term_memory.copy()
            memory_to_save["topics_discussed"] = dict(memory_to_save["topics_discussed"])
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error saving memory: {e}")
    
    def clear_all_memory(self):
        """Clear both short-term and long-term memory"""
        self.short_term_memory = []
        self.long_term_memory = {
            "user_preferences": {},
            "important_facts": [],
            "topics_discussed": defaultdict(int),
            "user_context": {},
            "key_conversations": []
        }
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        print("🧠 All memory cleared")
    
    def export_memory(self, filename: str = "memory_export.json"):
        """Export all memory to a file"""
        export_data = {
            "short_term_memory": self.short_term_memory,
            "long_term_memory": {
                **self.long_term_memory,
                "topics_discussed": dict(self.long_term_memory["topics_discussed"])
            },
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Memory exported to {filename}")
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage"""
        return {
            "short_term_messages": len(self.short_term_memory),
            "long_term_facts": len(self.long_term_memory["important_facts"]),
            "user_preferences": len(self.long_term_memory["user_preferences"]),
            "topics_tracked": len(self.long_term_memory["topics_discussed"]),
            "key_conversations": len(self.long_term_memory["key_conversations"]),
            "total_topic_mentions": sum(self.long_term_memory["topics_discussed"].values()),
            "memory_file_exists": os.path.exists(self.memory_file)
        }
        
