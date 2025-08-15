#!/usr/bin/env python3
"""
Example of how the main AI agent can integrate with the embedding system
Shows practical usage patterns for RAG (Retrieval-Augmented Generation)
"""

import requests
import json
from typing import List, Dict, Optional

class EmbeddingClient:
    """Simple client for the main AI agent to query embeddings"""
    
    def __init__(self, 
                 base_url: str = "http://127.0.0.1:8000",
                 user_id: str = None,
                 machine_id: str = None, 
                 project_root: str = None,
                 auth_token: str = None,
                 api_key: str = None):
        self.base_url = base_url
        self.user_id = user_id
        self.machine_id = machine_id
        self.project_root = project_root
        self.auth_token = auth_token
        self.api_key = api_key
    
    def search_relevant_files(self, 
                            query: str, 
                            k: int = 5, 
                            modality_filter: Optional[str] = None,
                            min_similarity: float = 0.7) -> List[Dict]:
        """
        Search for files relevant to a query
        
        Args:
            query: Search query (e.g., "player movement", "enemy sprites")
            k: Number of results to return
            modality_filter: Optional filter ('text', 'image', 'audio')
            min_similarity: Minimum similarity score (0-1)
        
        Returns:
            List of relevant files with similarity scores
        """
        try:
            data = {
                "action": "search",
                "query": query,
                "k": k,
                "user_id": self.user_id,
                "machine_id": self.machine_id,
                "project_root": self.project_root
            }
            
            if modality_filter:
                data["modality_filter"] = modality_filter
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
                "X-API-Key": self.api_key
            }
            
            response = requests.post(
                f"{self.base_url}/embed",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                results = result.get('results', [])
                
                # Filter by minimum similarity
                filtered_results = [
                    r for r in results 
                    if r['similarity'] >= min_similarity
                ]
                
                return filtered_results
            else:
                print(f"Embedding search failed: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return []
    
    def get_project_context(self, query: str) -> str:
        """
        Get relevant project context for a query (for RAG)
        
        Args:
            query: User's question or request
            
        Returns:
            Formatted context string with relevant files
        """
        results = self.search_relevant_files(query, k=10, min_similarity=0.6)
        
        if not results:
            return "No relevant project files found."
        
        context_parts = ["📁 Relevant project files for your query:"]
        
        # Group by modality
        by_modality = {}
        for item in results:
            modality = item['modality']
            if modality not in by_modality:
                by_modality[modality] = []
            by_modality[modality].append(item)
        
        # Format by type
        if 'text' in by_modality:
            context_parts.append("\n🔤 Code/Text files:")
            for item in by_modality['text'][:5]:  # Top 5
                context_parts.append(f"  • {item['file_path']} (relevance: {item['similarity']:.1%})")
        
        if 'image' in by_modality:
            context_parts.append("\n🎨 Image files:")
            for item in by_modality['image'][:3]:  # Top 3
                context_parts.append(f"  • {item['file_path']} (relevance: {item['similarity']:.1%})")
        
        if 'audio' in by_modality:
            context_parts.append("\n🔊 Audio files:")
            for item in by_modality['audio'][:3]:  # Top 3
                context_parts.append(f"  • {item['file_path']} (relevance: {item['similarity']:.1%})")
        
        return "\n".join(context_parts)

# Example usage patterns for the main AI agent
class MainAgent:
    """Example main AI agent that uses embedding system"""
    
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client
    
    def answer_with_context(self, user_query: str) -> str:
        """
        Answer user query with relevant project context
        This is how RAG (Retrieval-Augmented Generation) works
        """
        
        # 1. Get relevant project files
        relevant_files = self.embedding_client.search_relevant_files(
            query=user_query,
            k=8,
            min_similarity=0.6
        )
        
        # 2. Format context for the LLM
        if relevant_files:
            context = self.embedding_client.get_project_context(user_query)
            
            # 3. Create enhanced prompt with context
            enhanced_prompt = f"""
User Question: {user_query}

{context}

Based on the relevant files above, please provide a helpful answer about the user's Godot project.
Focus on the specific files and assets that are most relevant to their question.
"""
            
            # 4. In a real implementation, you'd send this to your LLM
            # For this example, we'll just return the context
            return f"🤖 I found relevant project files for your question:\n\n{context}\n\n💡 This context would be sent to the LLM for a complete answer."
        
        else:
            return "🤖 I couldn't find any relevant files in your project for that question. Could you be more specific?"
    
    def suggest_relevant_assets(self, task_description: str) -> List[str]:
        """Suggest relevant assets for a development task"""
        
        # Search for different types of assets
        code_files = self.embedding_client.search_relevant_files(
            query=task_description,
            k=5,
            modality_filter="text",
            min_similarity=0.5
        )
        
        image_files = self.embedding_client.search_relevant_files(
            query=task_description,
            k=3,
            modality_filter="image", 
            min_similarity=0.5
        )
        
        suggestions = []
        
        if code_files:
            suggestions.append("📝 Relevant code files:")
            for file in code_files:
                suggestions.append(f"  • {file['file_path']}")
        
        if image_files:
            suggestions.append("🎨 Relevant image assets:")
            for file in image_files:
                suggestions.append(f"  • {file['file_path']}")
        
        return suggestions
    
    def find_similar_implementations(self, feature_description: str) -> List[str]:
        """Find existing similar implementations in the project"""
        
        results = self.embedding_client.search_relevant_files(
            query=f"implementation {feature_description}",
            k=5,
            modality_filter="text",
            min_similarity=0.6
        )
        
        if results:
            return [f"📋 {r['file_path']} ({r['similarity']:.1%} similar)" for r in results]
        else:
            return ["No similar implementations found in your project."]

# Example usage
def demo_usage():
    """Demonstrate how to use the embedding system"""
    
    print("🤖 Main AI Agent Integration Example")
    print("=" * 50)
    
    # Initialize embedding client (you'd get these from auth system)
    client = EmbeddingClient(
        user_id="106469680334583136136",
        machine_id="XP4191P2VD", 
        project_root="/Users/alikavoosi/Desktop/3d-design/game-from-scratch-project/new-game-project",
        auth_token="your_auth_token",
        api_key="your_api_key"
    )
    
    # Initialize main agent
    agent = MainAgent(client)
    
    # Example queries
    example_queries = [
        "How does player movement work in my game?",
        "Show me all enemy-related assets",
        "Find UI button graphics",
        "Where is the audio system implemented?",
        "What shaders are being used for effects?"
    ]
    
    print("🔍 Example queries and responses:")
    print()
    
    for query in example_queries:
        print(f"❓ User: {query}")
        
        # This would normally send the context to your LLM
        # For demo, we'll just show the relevant files
        relevant_files = client.search_relevant_files(query, k=3)
        
        if relevant_files:
            print("📁 Found relevant files:")
            for file in relevant_files:
                print(f"   • {file['file_path']} ({file['similarity']:.1%} relevant)")
        else:
            print("   No relevant files found.")
        
        print()
    
    print("💡 Integration Tips:")
    print("1. Use search_relevant_files() to get context for user queries")
    print("2. Include file paths and similarity scores in LLM prompts")
    print("3. Filter by modality for specific asset types")
    print("4. Set appropriate similarity thresholds (0.6-0.8 typically good)")
    print("5. Combine multiple searches for comprehensive context")

if __name__ == "__main__":
    demo_usage()