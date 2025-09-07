import os
import re
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import pickle
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    ADVANCED_RAG_AVAILABLE = True
except ImportError:
    ADVANCED_RAG_AVAILABLE = False
    print("⚠️ Advanced RAG not available. Install: pip install sentence-transformers faiss-cpu")

class RAGTool:
    """Enhanced RAG with LLM-powered answer generation for maximum accuracy"""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50, use_llm: bool = True):
        self.docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        
        # Optimized parameters - More strict thresholds for better fallback detection
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = 0.25  # Increased from 0.15 for stricter matching
        self.confidence_threshold = 0.3   # New threshold for confidence scoring
        
        # Initialize LLM client
        self.use_llm = use_llm
        self.groq_client = None
        if self.use_llm:
            try:
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
                    print("✅ Groq client initialized for enhanced answer generation")
                else:
                    print("⚠️ GROQ_API_KEY not found. Using fallback answer generation.")
                    self.use_llm = False
            except ImportError:
                print("⚠️ Groq not available. Install: pip install groq")
                self.use_llm = False
        
        if ADVANCED_RAG_AVAILABLE:
            self.use_advanced = True
            self.model_name = 'all-MiniLM-L6-v2'
            print(f"🚀 Loading optimized sentence transformer: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name)
            self.embedding_dim = 384
            self.faiss_index = None
            self.documents = []
            self._load_or_create_index()
        else:
            self.use_advanced = False
            self.documents = self._load_documents_basic()
    
    def _create_smart_chunks(self, text: str) -> List[Dict]:
        """Create optimized chunks with better content preservation"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_words = 0
        
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            
            # If this paragraph is too large, split it into sentences
            if paragraph_words > self.chunk_size:
                sentences = self._split_into_sentences(paragraph)
                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    
                    if current_words + sentence_words > self.chunk_size and current_chunk:
                        # Create chunk
                        chunk_text = ' '.join(current_chunk).strip()
                        if len(chunk_text) > 30:  # Only meaningful chunks
                            chunks.append({
                                'content': chunk_text,
                                'word_count': current_words,
                                'type': 'mixed'
                            })
                        
                        # Start new chunk with overlap
                        overlap_text = self._get_overlap_text(current_chunk)
                        current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                        current_words = len(' '.join(current_chunk).split())
                    else:
                        current_chunk.append(sentence)
                        current_words += sentence_words
            else:
                # Check if adding this paragraph exceeds chunk size
                if current_words + paragraph_words > self.chunk_size and current_chunk:
                    # Create chunk from current content
                    chunk_text = ' '.join(current_chunk).strip()
                    if len(chunk_text) > 30:
                        chunks.append({
                            'content': chunk_text,
                            'word_count': current_words,
                            'type': 'paragraph_complete'
                        })
                    
                    # Start new chunk with this paragraph
                    current_chunk = [paragraph]
                    current_words = paragraph_words
                else:
                    current_chunk.append(paragraph)
                    current_words += paragraph_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) > 30:
                chunks.append({
                    'content': chunk_text,
                    'word_count': current_words,
                    'type': 'final'
                })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_overlap_text(self, sentences: List[str]) -> str:
        """Get overlap text based on word count"""
        if not sentences:
            return ""
        
        overlap_words = 0
        overlap_sentences = []
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if overlap_words + sentence_words <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_words += sentence_words
            else:
                break
        
        return ' '.join(overlap_sentences)
    
    def _build_index(self):
        """Build optimized FAISS index"""
        self.documents = []
        all_embeddings = []
        
        if not os.path.exists(self.docs_dir):
            print(f"⚠️ Documents directory not found: {self.docs_dir}")
            return
        
        doc_metadata = {}
        total_chunks = 0
        
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.docs_dir, filename)
                doc_metadata[filename] = os.path.getmtime(filepath)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunks = self._create_smart_chunks(content)
                        
                        print(f"📚 {filename}: {len(chunks)} optimized chunks")
                        total_chunks += len(chunks)
                        
                        for i, chunk_info in enumerate(chunks):
                            chunk_text = chunk_info['content']
                            
                            # Create embedding
                            embedding = self.sentence_model.encode(
                                chunk_text, 
                                convert_to_tensor=False,
                                normalize_embeddings=True
                            )
                            
                            doc_info = {
                                'filename': filename,
                                'chunk_id': i,
                                'content': chunk_text,
                                'word_count': chunk_info['word_count'],
                                'chunk_type': chunk_info['type'],
                                'embedding': embedding
                            }
                            
                            self.documents.append(doc_info)
                            all_embeddings.append(embedding)
                            
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
        
        if not all_embeddings:
            print("⚠️ No documents to index")
            return
        
        # Create FAISS index with normalized embeddings
        embeddings_matrix = np.array(all_embeddings).astype('float32')
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings_matrix)
        
        print(f"✅ Built optimized FAISS index with {total_chunks} chunks")
        self._save_index(doc_metadata)
    
    def run_groq_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Send prompt to Groq Llama 3.3 70B and return response"""
        if not self.groq_client:
            return None
            
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,  # Lower temperature for more focused answers
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Groq API error: {e}")
            return None
    
    def check_relevance_with_llm(self, query: str, context: str) -> Dict[str, any]:
        """Use LLM to check if context is relevant to query"""
        if not self.groq_client:
            return {"relevant": False, "confidence": 0.0}
        
        relevance_prompt = f"""You are a relevance checker. Your task is to determine if the provided context contains information that can answer the user's question.

Question: {query}

Context: {context}

Instructions:
1. Analyze if the context contains information directly relevant to answering the question
2. Consider partial matches - if context has some related information, mark as relevant
3. If the context is about completely different topics, mark as not relevant
4. Respond with ONLY one of these formats:
   - RELEVANT: [brief reason why it's relevant]
   - NOT_RELEVANT: [brief reason why it's not relevant]

Response:"""

        try:
            response = self.run_groq_llm(relevance_prompt, max_tokens=100)
            if response:
                if response.upper().startswith('RELEVANT'):
                    return {"relevant": True, "confidence": 0.8, "reason": response}
                elif response.upper().startswith('NOT_RELEVANT'):
                    return {"relevant": False, "confidence": 0.9, "reason": response}
        except Exception as e:
            print(f"⚠️ Relevance check error: {e}")
        
        return {"relevant": False, "confidence": 0.0}
    
    def search_and_answer(self, query: str, top_k: int = 5) -> Dict:
        """Enhanced search with better fallback detection"""
        
        if not self.use_advanced:
            return self._search_basic(query)
        
        if not self.documents or self.faiss_index is None:
            return {"fallback": True, "message": "No documents in knowledge base.", "should_use_groq": True}
        
        try:
            # Create normalized query embedding
            query_embedding = self.sentence_model.encode(
                [query], 
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            query_embedding = query_embedding.astype('float32')
            
            # Search with larger k for better selection
            k = min(15, len(self.documents))
            similarities, indices = self.faiss_index.search(query_embedding, k)
            
            # Collect and rank results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    relevance_score = self._calculate_enhanced_relevance(query, doc, similarity)
                    
                    results.append({
                        'similarity': float(similarity),
                        'relevance': relevance_score,
                        'doc': doc,
                        'combined_score': self._compute_final_score(similarity, relevance_score, doc)
                    })
            
            # Sort by combined score
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            print(f"🔍 Search for: '{query}'")
            print(f"📋 Top {min(top_k, len(results))} matches found")
            
            # Enhanced relevance checking
            if not results:
                return {
                    "fallback": True, 
                    "message": f"No relevant information found for: {query}",
                    "should_use_groq": True
                }
            
            best_score = results[0]['combined_score']
            best_similarity = results[0]['similarity']
            
            # More strict thresholds for fallback detection
            if best_score < self.similarity_threshold or best_similarity < 0.2:
                print(f"⚠️ Low relevance scores - best_score: {best_score:.3f}, similarity: {best_similarity:.3f}")
                
                # Use LLM to double-check relevance if available
                if self.use_llm and self.groq_client:
                    top_context = results[0]['doc']['content']
                    relevance_check = self.check_relevance_with_llm(query, top_context)
                    
                    if not relevance_check.get('relevant', False):
                        print(f"🤖 LLM confirms low relevance: {relevance_check.get('reason', 'No reason')}")
                        return {
                            "fallback": True, 
                            "message": f"Information about '{query}' not found in knowledge base.",
                            "should_use_groq": True,
                            "confidence": best_score,
                            "llm_check": relevance_check.get('reason', '')
                        }
                else:
                    # Without LLM, use stricter thresholds
                    return {
                        "fallback": True, 
                        "message": f"Information about '{query}' not found in knowledge base.",
                        "should_use_groq": True,
                        "confidence": best_score
                    }
            
            # Generate response using LLM if available, otherwise use fallback
            if self.use_llm and self.groq_client:
                response = self._generate_llm_response(query, results[:top_k])
                
                # Check if LLM response indicates it doesn't know
                if self._response_indicates_unknown(response):
                    return {
                        "fallback": True, 
                        "message": f"Information about '{query}' not found in knowledge base.",
                        "should_use_groq": True,
                        "confidence": best_score,
                        "llm_response": response
                    }
            else:
                response = self._generate_improved_response(query, results[:top_k])
            
            # Add sources
            sources = list(set(r['doc']['filename'] for r in results[:3]))
            if sources:
                response += f"\n\n📚 Sources: {', '.join(sources)}"
            
            return {
                "fallback": False, 
                "message": response,
                "confidence": best_score,
                "sources": sources,
                "method": "LLM-enhanced" if (self.use_llm and self.groq_client) else "rule-based",
                "should_use_groq": False
            }
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return {
                "fallback": True, 
                "message": f"Search error: {str(e)}",
                "should_use_groq": True
            }
    
    def _response_indicates_unknown(self, response: str) -> bool:
        """Check if LLM response indicates it doesn't know the answer"""
        if not response:
            return True
            
        unknown_indicators = [
            "not defined in the provided",
            "not available in the given context",
            "not provided in the context",
            "doesn't contain information",
            "no information about",
            "not mentioned in the",
            "not found in the provided",
            "cannot be answered based on",
            "not specified in the context",
            "information is not available"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in unknown_indicators)
    
    def _generate_llm_response(self, query: str, results: List[Dict]) -> str:
        """Generate response using LLM with retrieved context"""
        if not results:
            return "No relevant information found."
        
        # Prepare context from top results
        context_parts = []
        for i, result in enumerate(results[:5]):  # Use top 5 results for context
            content = result['doc']['content']
            context_parts.append(f"Context {i+1}: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for better unknown detection
        prompt = f"""You are a knowledgeable assistant. Answer the user's question using ONLY the provided context. If the context doesn't contain relevant information to answer the question, clearly state that the information is not available.

Question: {query}

Context:
{context}

Instructions:
1. ONLY answer if the context contains relevant information about the question
2. If the context is about different topics or doesn't contain the answer, respond with: "The information about [topic] is not available in the provided context."
3. If you can partially answer, provide what information you can but note limitations
4. Be accurate and don't make up information not in the context
5. Keep responses focused and relevant
6. Maximum 3-4 sentences unless more detail is clearly needed

Answer:"""

        # Get response from LLM
        llm_response = self.run_groq_llm(prompt, max_tokens=300)
        
        if llm_response:
            return llm_response
        else:
            # Fallback to rule-based response if LLM fails
            print("⚠️ LLM failed, using fallback response generation")
            return self._generate_improved_response(query, results)
    
    def _calculate_enhanced_relevance(self, query: str, doc: Dict, similarity: float) -> float:
        """Enhanced relevance calculation with better thresholds"""
        content = doc['content'].lower()
        query_lower = query.lower()
        
        # Keyword overlap bonus
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        content_words = set(re.findall(r'\b\w+\b', content))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        query_words = query_words - stop_words
        content_words = content_words - stop_words
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        overlap_score = (overlap / len(query_words)) if query_words else 0
        
        # Penalize if there's very little overlap
        if overlap_score < 0.2:  # Less than 20% word overlap
            overlap_score *= 0.5
        
        # Exact phrase matching
        phrase_bonus = 0.3 if query_lower in content else 0
        
        # Definition pattern bonus
        if any(pattern in content for pattern in ['definition:', 'refers to', 'is a', 'is the', 'means']):
            phrase_bonus += 0.2
        
        # Chunk quality bonus
        quality_bonus = 0.1 if doc.get('chunk_type') == 'paragraph_complete' else 0
        
        # Length penalty for very short or very long chunks
        word_count = doc['word_count']
        length_penalty = 0.2 if word_count < 50 else (0.1 if word_count > 400 else 0)
        
        relevance = overlap_score + phrase_bonus + quality_bonus - length_penalty
        return max(0, min(1, relevance))
    
    def _compute_final_score(self, similarity: float, relevance: float, doc: Dict) -> float:
        """Compute final ranking score with stricter criteria"""
        base_score = (similarity * 0.7) + (relevance * 0.3)
        
        # Content quality multiplier
        content = doc['content'].lower()
        
        # Boost for definition-like content
        if any(pattern in content for pattern in ['definition:', 'refers to', 'is a', 'is the', 'means']):
            base_score *= 1.2
        
        # Boost for structured content
        if ':' in content or any(marker in content for marker in ['1.', '2.', '-', '•']):
            base_score *= 1.1
        
        # Penalize very low similarity scores more heavily
        if similarity < 0.3:
            base_score *= 0.7
            
        return min(1.0, base_score)
    
    def _generate_improved_response(self, query: str, results: List[Dict]) -> str:
        """Fallback response generation (rule-based) with better unknown detection"""
        if not results:
            return "No relevant information found."
        
        query_lower = query.lower()
        best_content = results[0]['doc']['content']
        
        # Check if the content is actually relevant
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        content_words = set(re.findall(r'\b\w+\b', best_content.lower()))
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        query_words = query_words - stop_words
        content_words = content_words - stop_words
        
        if query_words and len(query_words & content_words) / len(query_words) < 0.15:
            return f"Information about '{query}' is not available in the knowledge base."
        
        # Strategy for "what is" questions
        if query_lower.startswith(('what is', 'what are', 'define')):
            for result in results:
                content = result['doc']['content']
                if any(pattern in content.lower() for pattern in [
                    'definition:', 'refers to', 'is a', 'is the', 'means'
                ]):
                    return self._clean_response(content)
            
            # Extract definition from best match
            return self._extract_definition_from_content(best_content, query)
        
        # For other questions, use best match
        return self._clean_response(best_content)
    
    def _extract_definition_from_content(self, content: str, query: str) -> str:
        """Extract definition-like information from content"""
        sentences = re.split(r'[.!?]+', content)
        query_terms = query.lower().replace('what is ', '').replace('what are ', '').strip().split()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_lower = sentence.lower()
            
            if (any(term in sentence_lower for term in query_terms) and 
                any(pattern in sentence_lower for pattern in ['is a', 'is the', 'refers to', 'means'])):
                return sentence + '.'
        
        # Return first meaningful sentence if definition pattern not found
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                return sentence + '.'
        
        return content[:200] + '...' if len(content) > 200 else content
    
    def _clean_response(self, content: str) -> str:
        """Clean and format the response"""
        content = content.strip()
        content = re.sub(r'\s+', ' ', content)
        
        if not content.endswith(('.', '!', '?')):
            content += '.'
        
        # Truncate if too long while preserving sentence boundaries
        if len(content) > 600:
            sentences = re.split(r'[.!?]+', content)
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if current_length + len(sentence) < 500:
                    truncated.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            if truncated:
                content = '. '.join(truncated) + '.'
        
        return content
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        if not self.documents:
            return {"error": "No documents loaded"}
        
        word_counts = [doc['word_count'] for doc in self.documents]
        
        return {
            "total_chunks": len(self.documents),
            "total_files": len(set(doc['filename'] for doc in self.documents)),
            "avg_chunk_words": np.mean(word_counts),
            "min_chunk_words": min(word_counts),
            "max_chunk_words": max(word_counts),
            "chunk_size_setting": self.chunk_size,
            "overlap_setting": self.overlap,
            "advanced_mode": self.use_advanced,
            "llm_enabled": self.use_llm and self.groq_client is not None,
            "similarity_threshold": self.similarity_threshold,
            "confidence_threshold": self.confidence_threshold
        }
    
    # Cache management methods (keeping existing implementation)
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        index_path = os.path.join(self.cache_dir, 'faiss_index.bin')
        docs_path = os.path.join(self.cache_dir, 'documents.pkl')
        metadata_path = os.path.join(self.cache_dir, 'metadata.json')
        
        need_rebuild = self._need_rebuild_index(metadata_path)
        
        if not need_rebuild and os.path.exists(index_path) and os.path.exists(docs_path):
            print("📚 Loading existing FAISS index...")
            try:
                self.faiss_index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"✅ Loaded {len(self.documents)} chunks from cache")
                return
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}. Rebuilding...")
        
        print("🔨 Building new optimized FAISS index...")
        self._build_index()
    
    def _need_rebuild_index(self, metadata_path: str) -> bool:
        """Check if index needs rebuilding"""
        if not os.path.exists(metadata_path):
            return True
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            current_docs = {}
            if os.path.exists(self.docs_dir):
                for filename in os.listdir(self.docs_dir):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(self.docs_dir, filename)
                        current_docs[filename] = os.path.getmtime(filepath)
            
            return (current_docs != metadata.get('documents', {}) or
                    self.chunk_size != metadata.get('chunk_size', 300) or
                    self.overlap != metadata.get('overlap', 50))
        except:
            return True
    
    def _save_index(self, doc_metadata: Dict):
        """Save FAISS index and documents to cache"""
        try:
            index_path = os.path.join(self.cache_dir, 'faiss_index.bin')
            docs_path = os.path.join(self.cache_dir, 'documents.pkl')
            metadata_path = os.path.join(self.cache_dir, 'metadata.json')
            
            faiss.write_index(self.faiss_index, index_path)
            
            docs_to_save = []
            for doc in self.documents:
                doc_copy = doc.copy()
                doc_copy.pop('embedding', None)
                docs_to_save.append(doc_copy)
            
            with open(docs_path, 'wb') as f:
                pickle.dump(docs_to_save, f)
            
            metadata = {
                'documents': doc_metadata,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            print("💾 Enhanced index saved to cache")
            
        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")
    
    def _load_documents_basic(self) -> List[Dict[str, str]]:
        documents = []
        
        if not os.path.exists(self.docs_dir):
            print(f"⚠️ Documents directory not found: {self.docs_dir}")
            return documents
        
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.docs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunks = self._create_smart_chunks(content)
                        
                        for i, chunk_info in enumerate(chunks):
                            documents.append({
                                'filename': filename,
                                'chunk_id': i,
                                'content': chunk_info['content'],
                                'word_count': chunk_info['word_count'],
                                'chunk_type': chunk_info['type']
                            })
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
        
        print(f"📚 Loaded {len(documents)} chunks in basic mode")
        return documents

    def _search_basic(self, query: str) -> Dict:
        """Basic search without FAISS (fallback mode)"""
        if not self.documents:
            return {
                "fallback": True, 
                "message": "No documents in knowledge base.", 
                "should_use_groq": True
            }
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        query_words = query_words - stop_words
        
        if not query_words:
            return {
                "fallback": True, 
                "message": "Query too general.", 
                "should_use_groq": True
            }
        
        # Score documents by word overlap
        scored_docs = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            content_words = set(re.findall(r'\b\w+\b', content_lower))
            
            overlap = len(query_words & content_words)
            if overlap > 0:
                overlap_score = overlap / len(query_words)
                
                # Bonus for exact phrase match
                phrase_bonus = 0.3 if query_lower in content_lower else 0
                
                # Bonus for definition patterns
                definition_bonus = 0.2 if any(pattern in content_lower for pattern in ['definition:', 'refers to', 'is a', 'is the', 'means']) else 0
                
                total_score = overlap_score + phrase_bonus + definition_bonus
                
                scored_docs.append({
                    'doc': doc,
                    'score': total_score,
                    'overlap': overlap
                })
        
        if not scored_docs:
            return {
                "fallback": True, 
                "message": f"Information about '{query}' not found in knowledge base.", 
                "should_use_groq": True
            }
        
        # Sort by score
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        best_score = scored_docs[0]['score']
        
        # Check if score is too low
        if best_score < 0.15:  # Adjust threshold as needed
            return {
                "fallback": True, 
                "message": f"Information about '{query}' not found in knowledge base.", 
                "should_use_groq": True,
                "confidence": best_score
            }
        
        # Generate response from best matches
        best_docs = scored_docs[:3]  # Use top 3
        response = self._generate_improved_response(query, [{'doc': d['doc']} for d in best_docs])
        
        # Get sources
        sources = list(set(d['doc']['filename'] for d in best_docs))
        if sources:
            response += f"\n\n📚 Sources: {', '.join(sources)}"
        
        return {
            "fallback": False, 
            "message": response,
            "confidence": best_score,
            "sources": sources,
            "method": "basic-search",
            "should_use_groq": False
        }