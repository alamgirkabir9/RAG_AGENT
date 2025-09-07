import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime
import logging
import traceback
import time
from functools import wraps
from agent.agent import RAGAgent

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG Agent (singleton pattern)
rag_agent = None

def get_agent():
    """Get or create RAG agent instance"""
    global rag_agent
    if rag_agent is None:
        rag_agent = RAGAgent()
    return rag_agent

def measure_response_time(f):
    """Decorator to measure and log response time"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
        
        # Log the response time
        logger.info(f"{request.endpoint} - Response time: {response_time}ms")
        
        # Add response time to JSON response if it's a JSON response
        if hasattr(result, 'json') or isinstance(result, tuple):
            if isinstance(result, tuple):
                response_data, status_code = result
            else:
                response_data = result
                status_code = 200
                
            if hasattr(response_data, 'json'):
                json_data = response_data.json
                json_data['response_time_ms'] = response_time
                return jsonify(json_data), status_code
        
        return result
    return decorated_function

# HTML template for web interface (updated to show response times)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Agent API</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .chat-container { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background: white; margin-bottom: 10px; }
        .message { margin: 10px 0; }
        .user { text-align: right; }
        .user .content { background: #007bff; color: white; display: inline-block; padding: 8px 12px; border-radius: 15px; max-width: 70%; }
        .agent .content { background: #e9ecef; display: inline-block; padding: 8px 12px; border-radius: 15px; max-width: 70%; }
        .response-info { font-size: 0.8em; color: #666; margin-top: 5px; }
        input[type="text"] { width: calc(100% - 120px); padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 20px; }
        .stat-item { background: white; padding: 15px; border-radius: 8px; text-align: center; }
        .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }
        pre { background: #f8f9fa; padding: 10px; overflow-x: auto; }
        .performance-indicator { 
            display: inline-block; 
            padding: 2px 6px; 
            border-radius: 3px; 
            font-size: 0.7em; 
            font-weight: bold; 
        }
        .fast { background: #d4edda; color: #155724; }
        .medium { background: #fff3cd; color: #856404; }
        .slow { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>🤖 RAG Agent API</h1>
    
    <div class="container">
        <h2>Interactive Chat</h2>
        <div id="chat-container" class="chat-container"></div>
        <input type="text" id="query-input" placeholder="Ask me anything..." onkeypress="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()">Send</button>
        <button onclick="clearChat()">Clear</button>
    </div>

    <div class="container">
        <h2>API Endpoints</h2>
        
        <div class="endpoint">
            <h3>POST /api/query</h3>
            <p>Send a query to the RAG agent</p>
            <pre>{
  "query": "What is artificial intelligence?",
  "include_context": true
}</pre>
        </div>

        <div class="endpoint">
            <h3>GET /api/stats</h3>
            <p>Get agent statistics and status</p>
        </div>

        <div class="endpoint">
            <h3>GET /api/memory</h3>
            <p>Get conversation memory summary</p>
        </div>

        <div class="endpoint">
            <h3>POST /api/memory/clear</h3>
            <p>Clear conversation memory</p>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('query-input');
            const query = input.value.trim();
            if (!query) return;

            const chatContainer = document.getElementById('chat-container');
            
            // Add user message
            addMessage('user', query);
            input.value = '';

            const startTime = performance.now();

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, include_context: true })
                });

                const data = await response.json();
                const endTime = performance.now();
                const clientResponseTime = Math.round(endTime - startTime);
                
                if (data.success) {
                    addMessage('agent', data.response, {
                        server_time: data.response_time_ms,
                        client_time: clientResponseTime,
                        timestamp: data.timestamp
                    });
                } else {
                    addMessage('agent', `Error: ${data.error}`, {
                        server_time: data.response_time_ms,
                        client_time: clientResponseTime
                    });
                }
            } catch (error) {
                const endTime = performance.now();
                const clientResponseTime = Math.round(endTime - startTime);
                addMessage('agent', `Error: ${error.message}`, {
                    client_time: clientResponseTime
                });
            }

            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function addMessage(role, content, timing = {}) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            let responseInfo = '';
            if (timing.server_time || timing.client_time) {
                const serverTime = timing.server_time || 'N/A';
                const clientTime = timing.client_time || 'N/A';
                
                // Performance indicator based on server response time
                let perfClass = 'fast';
                if (timing.server_time > 2000) perfClass = 'slow';
                else if (timing.server_time > 1000) perfClass = 'medium';
                
                responseInfo = `
                    <div class="response-info">
                        <span class="performance-indicator ${perfClass}">Server: ${serverTime}ms</span>
                        <span style="margin-left: 10px;">Client: ${clientTime}ms</span>
                        ${timing.timestamp ? `<span style="margin-left: 10px;">${new Date(timing.timestamp).toLocaleTimeString()}</span>` : ''}
                    </div>
                `;
            }
            
            messageDiv.innerHTML = `<div class="content">${content}${responseInfo}</div>`;
            chatContainer.appendChild(messageDiv);
        }

        function clearChat() {
            document.getElementById('chat-container').innerHTML = '';
            fetch('/api/memory/clear', { method: 'POST' });
        }

        // Load stats on page load
        window.onload = function() {
            const startTime = performance.now();
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    const endTime = performance.now();
                    console.log('Agent Stats:', data);
                    console.log(`Stats loaded in ${Math.round(endTime - startTime)}ms`);
                });
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/query', methods=['POST'])
@measure_response_time
def api_query():
    """Process a query through the RAG agent"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400

        query = data['query'].strip()
        include_context = data.get('include_context', False)
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Empty query'
            }), 400

        # Process query through RAG agent
        agent_start_time = time.time()
        agent = get_agent()
        response = agent.process_query(query)
        agent_end_time = time.time()
        agent_processing_time = round((agent_end_time - agent_start_time) * 1000, 2)
        
        # Get additional context if requested
        context_info = {}
        context_start_time = time.time()
        if include_context:
            try:
                # Get memory context
                context_info['memory_stats'] = agent.memory.get_memory_stats()
                context_info['recent_context'] = agent.memory.get_recent_context(3)
                
                # Get RAG stats if available
                if hasattr(agent, 'rag_tool'):
                    context_info['rag_stats'] = agent.rag_tool.get_stats()
                    
            except Exception as e:
                logger.warning(f"Could not get context info: {e}")
        context_end_time = time.time()
        context_processing_time = round((context_end_time - context_start_time) * 1000, 2)

        return jsonify({
            'success': True,
            'response': response,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'context': context_info if include_context else None,
            'performance': {
                'agent_processing_time_ms': agent_processing_time,
                'context_processing_time_ms': context_processing_time if include_context else 0
            }
        })

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
@measure_response_time
def api_stats():
    """Get agent statistics"""
    try:
        agent = get_agent()
        stats = {}
        
        # Memory statistics
        try:
            memory_start = time.time()
            stats['memory'] = agent.memory.get_memory_stats()
            memory_time = round((time.time() - memory_start) * 1000, 2)
        except Exception as e:
            stats['memory'] = {'error': str(e)}
            memory_time = 0
        
        # RAG statistics
        try:
            rag_start = time.time()
            if hasattr(agent, 'rag_tool'):
                stats['rag'] = agent.rag_tool.get_stats()
            else:
                stats['rag'] = {'status': 'not_available'}
            rag_time = round((time.time() - rag_start) * 1000, 2)
        except Exception as e:
            stats['rag'] = {'error': str(e)}
            rag_time = 0
        
        # System information
        stats['system'] = {
            'python_version': sys.version,
            'agent_initialized': agent is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Performance breakdown
        stats['performance'] = {
            'memory_query_time_ms': memory_time,
            'rag_query_time_ms': rag_time
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/memory', methods=['GET'])
@measure_response_time
def api_memory():
    """Get memory information"""
    try:
        agent = get_agent()
        
        # Time each memory operation
        stats_start = time.time()
        memory_stats = agent.memory.get_memory_stats()
        stats_time = round((time.time() - stats_start) * 1000, 2)
        
        summary_start = time.time()
        user_summary = agent.memory.get_user_summary()
        summary_time = round((time.time() - summary_start) * 1000, 2)
        
        context_start = time.time()
        recent_context = agent.memory.get_recent_context(5)
        context_time = round((time.time() - context_start) * 1000, 2)
        
        memory_info = {
            'stats': memory_stats,
            'user_summary': user_summary,
            'recent_context': recent_context
        }
        
        return jsonify({
            'success': True,
            'memory': memory_info,
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'stats_time_ms': stats_time,
                'summary_time_ms': summary_time,
                'context_time_ms': context_time
            }
        })

    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/memory/clear', methods=['POST'])
@measure_response_time
def api_clear_memory():
    """Clear conversation memory"""
    try:
        agent = get_agent()
        clear_start = time.time()
        agent.memory.clear_all_memory()
        clear_time = round((time.time() - clear_start) * 1000, 2)
        
        return jsonify({
            'success': True,
            'message': 'Memory cleared successfully',
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'clear_time_ms': clear_time
            }
        })

    except Exception as e:
        logger.error(f"Memory clear error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/memory/search', methods=['POST'])
@measure_response_time
def api_search_memory():
    """Search through memory"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400

        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Empty query'
            }), 400

        agent = get_agent()
        search_start = time.time()
        search_results = agent.memory.search_memory(query)
        search_time = round((time.time() - search_start) * 1000, 2)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': search_results,
            'count': len(search_results),
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'search_time_ms': search_time
            }
        })

    except Exception as e:
        logger.error(f"Memory search error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
@measure_response_time
def health_check():
    """Health check endpoint"""
    try:
        health_start = time.time()
        agent = get_agent()
        health_time = round((time.time() - health_start) * 1000, 2)
        
        return jsonify({
            'status': 'healthy',
            'agent_ready': agent is not None,
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'health_check_time_ms': health_time
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

def main():
    """Run the API server"""
    print("🚀 Starting RAG Agent API Server...")
    print("📱 Web Interface: http://localhost:5000")
    print("🔗 API Base URL: http://localhost:5000/api")
    print("📊 Health Check: http://localhost:5000/api/health")
    print("⏱️  Response times will be tracked for all endpoints")
    print("Press Ctrl+C to stop\n")
    
    # Initialize agent on startup
    try:
        init_start = time.time()
        get_agent()
        init_time = round((time.time() - init_start) * 1000, 2)
        print(f"✅ RAG Agent initialized successfully in {init_time}ms")
    except Exception as e:
        print(f"⚠️ Warning: RAG Agent initialization failed: {e}")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()