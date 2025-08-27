"""
© 2025 Simplifine Corp. Original backend contribution for this Godot fork.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.

Cloud-native conversation memory management with AI-powered summarization using Weaviate.
Designed for Google Cloud Run deployment and frontend-managed conversations.
"""
import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Any
import litellm
from litellm import acompletion
from memory_config import MemoryConfig
from datetime import datetime, timezone

class ConversationMemoryManager:
    """
    Cloud-native conversation memory management using Weaviate for storage.
    Designed to work with frontend-managed conversations in Google Cloud Run.
    """
    
    def __init__(self, weaviate_manager=None):
        self.weaviate_manager = weaviate_manager  # Injected from app.py
        
        # Get configuration values
        self.summarization_models = MemoryConfig.get_summarization_models()
        self.TOKEN_LIMITS = MemoryConfig.TOKEN_LIMITS
        self.enabled = MemoryConfig.is_enabled()
        
        # Initialize Weaviate collections if manager provided
        if self.weaviate_manager and self.enabled:
            self._init_conversation_collections()
    
    def _init_conversation_collections(self):
        """Initialize Weaviate collections for conversation summaries"""
        if not self.weaviate_manager:
            return
            
        try:
            collections = self.weaviate_manager.client.collections
            
            # Conversation summaries collection
            if not collections.exists("ConversationSummary"):
                from weaviate.classes.config import Configure, Property, DataType, VectorDistances
                collections.create(
                    name="ConversationSummary",
                    properties=[
                        Property(name="user_id", data_type=DataType.TEXT),
                        Property(name="conversation_hash", data_type=DataType.TEXT),
                        Property(name="fuzzy_hash", data_type=DataType.TEXT),  # For edited message matching
                        Property(name="summary", data_type=DataType.TEXT),
                        Property(name="original_message_count", data_type=DataType.INT),
                        Property(name="summary_tokens", data_type=DataType.INT),
                        Property(name="model_used", data_type=DataType.TEXT),
                        Property(name="created_at", data_type=DataType.DATE),
                        Property(name="conversation_topic", data_type=DataType.TEXT),
                        Property(name="is_updated", data_type=DataType.BOOL),  # Track if this is an updated summary
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),  # We provide our own embeddings
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE,
                        ef_construction=128,
                        ef=64,
                        max_connections=16,
                    ),
                )
                print("✅ Created ConversationSummary collection in Weaviate")
                
        except Exception as e:
            print(f"❌ Failed to initialize conversation collections: {e}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximately 4 characters per token)"""
        return len(text) // 4
    
    def hash_messages(self, messages: List[Dict]) -> str:
        """Create a hash of messages for deduplication"""
        content = json.dumps([msg.get('content', '') for msg in messages], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _create_fuzzy_hash(self, messages: List[Dict]) -> str:
        """Create a fuzzy hash based on structure rather than exact content"""
        # Use message count + roles + first/last partial content for fuzzy matching
        if not messages:
            return ""
        
        structure_data = {
            'count': len(messages),
            'roles': [msg.get('role', '') for msg in messages],
            'first_snippet': messages[0].get('content', '')[:100] if messages else '',
            'last_snippet': messages[-1].get('content', '')[:100] if len(messages) > 1 else '',
        }
        
        structure_str = json.dumps(structure_data, sort_keys=True)
        return hashlib.md5(structure_str.encode()).hexdigest()
    
    async def create_summary(self, messages: List[Dict], user_id: str = "unknown") -> str:
        """
        Create an intelligent summary of conversation messages using AI
        """
        if not messages:
            return ""
        
        # Prepare content for summarization
        conversation_text = self._format_messages_for_summary(messages)
        
        # Create summarization prompt from configuration
        summary_prompt = MemoryConfig.get_summary_prompt_template().format(
            conversation_text=conversation_text
        )

        # Try summarization models in order of preference
        for model in self.summarization_models:
            try:
                print(f"MEMORY_SUMMARY: Attempting summarization with {model}")
                
                response = await acompletion(
                    model=model,
                    messages=[
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=MemoryConfig.SUMMARY_MAX_TOKENS,
                    temperature=MemoryConfig.SUMMARY_TEMPERATURE,
                    timeout=MemoryConfig.SUMMARY_TIMEOUT
                )
                
                summary = response.choices[0].message.content.strip()
                
                if summary:
                    print(f"MEMORY_SUMMARY: Successfully created summary with {model} ({len(summary)} chars)")
                    
                    # Store summary in Weaviate
                    await self._store_summary(user_id, messages, summary, model)
                    
                    return summary
                    
            except Exception as e:
                print(f"MEMORY_SUMMARY: Failed with {model}: {e}")
                continue
        
        # Fallback: Create a simple structured summary
        print("MEMORY_SUMMARY: AI summarization failed, creating fallback summary")
        return self._create_fallback_summary(messages)
    
    def _format_messages_for_summary(self, messages: List[Dict]) -> str:
        """Format messages for AI summarization"""
        formatted_lines = []
        
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))
            
            # Truncate very long content to avoid overwhelming the summarizer
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            
            formatted_lines.append(f"[{i+1}] {role.upper()}: {content}")
        
        return "\n".join(formatted_lines)
    
    def _create_fallback_summary(self, messages: List[Dict]) -> str:
        """Create a simple fallback summary without AI"""
        topics = []
        tools_used = []
        files_mentioned = []
        
        for msg in messages:
            content = str(msg.get('content', ''))
            
            # Extract tool usage
            if 'tool_calls' in msg or 'function_calls' in msg:
                tools_used.append("tool execution")
            
            # Extract file paths
            if 'res://' in content or '.gd' in content or '.tscn' in content:
                files_mentioned.append("file operations")
            
            # Extract common topics
            if any(keyword in content.lower() for keyword in ['error', 'fix', 'problem']):
                topics.append("debugging/fixes")
            if any(keyword in content.lower() for keyword in ['node', 'scene', 'script']):
                topics.append("scene/node work")
            if any(keyword in content.lower() for keyword in ['asset', 'install', 'addon']):
                topics.append("asset management")
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Topics covered: {', '.join(set(topics))}")
        if tools_used:
            summary_parts.append(f"Tools used: {len(set(tools_used))} different operations")
        if files_mentioned:
            summary_parts.append("File operations performed")
        
        summary_parts.append(f"Conversation included {len(messages)} messages")
        
        return f"[AUTO-SUMMARY] {'. '.join(summary_parts)}."
    
    async def _store_summary(self, user_id: str, messages: List[Dict], summary: str, model_used: str):
        """Store summary in Weaviate"""
        if not self.weaviate_manager:
            print("MEMORY_STORE: No Weaviate manager available")
            return
            
        try:
            conversation_hash = self.hash_messages(messages)
            fuzzy_hash = self._create_fuzzy_hash(messages)
            summary_tokens = self.estimate_tokens(summary)
            
            # Generate embedding for the summary for semantic search
            if hasattr(self.weaviate_manager, 'openai_client'):
                try:
                    embedding_response = self.weaviate_manager.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=summary
                    )
                    summary_embedding = embedding_response.data[0].embedding
                except Exception as e:
                    print(f"MEMORY_EMBEDDING: Failed to generate embedding: {e}")
                    summary_embedding = None
            else:
                summary_embedding = None
            
            # Extract topic from summary for better organization
            conversation_topic = self._extract_topic_from_summary(summary)
            
            collection = self.weaviate_manager.client.collections.get("ConversationSummary")
            collection.data.insert(
                properties={
                    "user_id": user_id,
                    "conversation_hash": conversation_hash,
                    "fuzzy_hash": fuzzy_hash,
                    "summary": summary,
                    "original_message_count": len(messages),
                    "summary_tokens": summary_tokens,
                    "model_used": model_used,
                    "created_at": datetime.now(timezone.utc),
                    "conversation_topic": conversation_topic,
                    "is_updated": False  # Initial summary
                },
                vector=summary_embedding
            )
            
            print(f"MEMORY_STORE: Stored summary in Weaviate ({summary_tokens} tokens, topic: {conversation_topic})")
                
        except Exception as e:
            print(f"MEMORY_STORE: Failed to store summary: {e}")
    
    def get_summary(self, user_id: str, conversation_hash: str) -> Optional[str]:
        """Retrieve existing summary if available"""
        if not self.weaviate_manager:
            return None
            
        try:
            from weaviate.classes.query import Filter
            
            collection = self.weaviate_manager.client.collections.get("ConversationSummary")
            
            # First try exact hash match
            result = collection.query.fetch_objects(
                where=Filter.by_property("user_id").equal(user_id) &
                      Filter.by_property("conversation_hash").equal(conversation_hash),
                limit=1,
                sort=collection.query.Sort.by_property("created_at", ascending=False)
            )
            
            if result.objects:
                return result.objects[0].properties["summary"]
            return None
                
        except Exception as e:
            print(f"MEMORY_RETRIEVE: Failed to get summary: {e}")
            return None
    
    def get_summary_with_fuzzy_fallback(self, user_id: str, messages: List[Dict]) -> Optional[str]:
        """Get summary with fallback for edited conversations"""
        if not self.weaviate_manager:
            return None
            
        # Try exact hash first
        exact_hash = self.hash_messages(messages)
        summary = self.get_summary(user_id, exact_hash)
        if summary:
            print("MEMORY_RETRIEVE: Found exact hash match")
            return summary
        
        # Try fuzzy hash for edited conversations
        try:
            from weaviate.classes.query import Filter
            
            fuzzy_hash = self._create_fuzzy_hash(messages)
            collection = self.weaviate_manager.client.collections.get("ConversationSummary")
            
            # First try fuzzy_hash match (for edited messages with similar structure)
            fuzzy_result = collection.query.fetch_objects(
                where=Filter.by_property("user_id").equal(user_id) &
                      Filter.by_property("fuzzy_hash").equal(fuzzy_hash),
                limit=1,
                sort=collection.query.Sort.by_property("created_at", ascending=False)
            )
            
            if fuzzy_result.objects:
                print("MEMORY_RETRIEVE: Found fuzzy hash match for edited messages")
                return fuzzy_result.objects[0].properties["summary"]
            
            # Fallback: search by structure similarity (message count ±2)
            similar_count_range = max(1, len(messages) - 2)  # Allow ±2 messages difference
            
            count_result = collection.query.fetch_objects(
                where=Filter.by_property("user_id").equal(user_id) &
                      Filter.by_property("original_message_count").greater_than(similar_count_range) &
                      Filter.by_property("original_message_count").less_than(len(messages) + 3),
                limit=5,
                sort=collection.query.Sort.by_property("created_at", ascending=False)
            )
            
            if count_result.objects:
                # Return the most recent similar summary
                print(f"MEMORY_RETRIEVE: Found count-based match with {count_result.objects[0].properties['original_message_count']} messages")
                return count_result.objects[0].properties["summary"]
                
        except Exception as e:
            print(f"MEMORY_RETRIEVE: Fuzzy search failed: {e}")
        
        return None
    
    async def manage_conversation_length(self, messages: List[Dict], model: str, user_id: str = "unknown") -> List[Dict]:
        """
        Manage conversation length with intelligent summarization
        """
        # If intelligent memory is disabled, return messages as-is
        if not self.enabled:
            print("MEMORY_MANAGE: Intelligent memory disabled, skipping management")
            return messages
            
        limit = MemoryConfig.get_token_limit(model)
        
        # Calculate current token usage
        total_tokens = sum(self.estimate_tokens(str(msg.get('content', ''))) for msg in messages)
        
        if total_tokens <= limit:
            return messages  # No pruning needed
        
        print(f"MEMORY_MANAGE: Token count {total_tokens} exceeds limit {limit}, managing conversation")
        
        # Don't prune very short conversations
        if len(messages) <= MemoryConfig.MIN_MESSAGES_TO_PRUNE:
            return messages
        
        # Identify sections to preserve and summarize
        system_msg = messages[0] if messages and messages[0].get('role') == 'system' else None
        recent_count = min(MemoryConfig.KEEP_RECENT_MESSAGES, len(messages) // 3)  # Keep more recent messages
        recent_messages = messages[-recent_count:]
        
        # Messages to summarize (middle section)
        start_idx = 1 if system_msg else 0
        end_idx = len(messages) - recent_count
        messages_to_summarize = messages[start_idx:end_idx]
        
        if messages_to_summarize:
            # Check if we already have a summary for this content (with fuzzy fallback for edited messages)
            existing_summary = self.get_summary_with_fuzzy_fallback(user_id, messages_to_summarize)
            
            if existing_summary:
                print(f"MEMORY_MANAGE: Using existing summary for {len(messages_to_summarize)} messages")
                summary_content = existing_summary
            else:
                print(f"MEMORY_MANAGE: Creating new summary for {len(messages_to_summarize)} messages")
                summary_content = await self.create_summary(messages_to_summarize, user_id)
        else:
            summary_content = "[No middle messages to summarize]"
        
        # Reconstruct conversation with summary
        managed_messages = []
        
        if system_msg:
            managed_messages.append(system_msg)
        
        # Add intelligent summary
        managed_messages.append({
            "role": "assistant", 
            "content": f"""[CONVERSATION CONTEXT SUMMARY]
Previous conversation context has been intelligently summarized to manage length.

{summary_content}

[END SUMMARY - Continuing with recent messages...]"""
        })
        
        # Add recent messages
        managed_messages.extend(recent_messages)
        
        # Verify final token count
        final_tokens = sum(self.estimate_tokens(str(msg.get('content', ''))) for msg in managed_messages)
        print(f"MEMORY_MANAGE: Reduced from {total_tokens} to {final_tokens} tokens ({len(messages)} to {len(managed_messages)} messages)")
        
        return managed_messages
    
    def cleanup_old_summaries(self, days_old: Optional[int] = None):
        """Clean up old summaries to prevent database bloat"""
        if not self.weaviate_manager:
            print("MEMORY_CLEANUP: No Weaviate manager available")
            return
            
        if days_old is None:
            days_old = MemoryConfig.CLEANUP_DAYS_DEFAULT
            
        try:
            from datetime import timedelta
            from weaviate.classes.query import Filter
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            collection = self.weaviate_manager.client.collections.get("ConversationSummary")
            
            # Query old summaries
            old_summaries = collection.query.fetch_objects(
                where=Filter.by_property("created_at").less_than(cutoff_date),
                limit=1000  # Batch delete
            )
            
            # Delete old summaries
            deleted_count = 0
            for obj in old_summaries.objects:
                collection.data.delete_by_id(obj.uuid)
                deleted_count += 1
            
            print(f"MEMORY_CLEANUP: Cleaned up {deleted_count} summaries older than {days_old} days")
                
        except Exception as e:
            print(f"MEMORY_CLEANUP: Failed to cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory management statistics"""
        if not self.weaviate_manager:
            return {"error": "No Weaviate manager available"}
            
        try:
            collection = self.weaviate_manager.client.collections.get("ConversationSummary")
            
            # Get total count
            total_result = collection.aggregate.over_all(total_count=True)
            total_summaries = total_result.total_count
            
            # Get model-specific stats (simplified for Weaviate)
            stats = {
                "total_summaries": total_summaries,
                "enabled": self.enabled,
                "weaviate_connected": True,
                "models_configured": self.summarization_models[:3],  # First 3
                "collection_exists": True
            }
            
            # Try to get recent summaries for more detailed stats
            try:
                recent_summaries = collection.query.fetch_objects(
                    limit=100,
                    sort=collection.query.Sort.by_property("created_at", ascending=False)
                )
                
                if recent_summaries.objects:
                    # Calculate averages from recent summaries
                    message_counts = [obj.properties.get("original_message_count", 0) for obj in recent_summaries.objects]
                    token_counts = [obj.properties.get("summary_tokens", 0) for obj in recent_summaries.objects]
                    
                    stats.update({
                        "avg_messages_summarized": round(sum(message_counts) / len(message_counts), 1) if message_counts else 0,
                        "avg_summary_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
                        "recent_summaries_sample": len(recent_summaries.objects)
                    })
            except Exception:
                pass  # Non-critical stats
                
            return stats
                
        except Exception as e:
            print(f"MEMORY_STATS: Failed to get stats: {e}")
            return {"error": str(e), "weaviate_connected": False}

    def _extract_topic_from_summary(self, summary: str) -> str:
        """Extract a topic/category from the summary for organization"""
        summary_lower = summary.lower()
        
        # Simple keyword-based topic extraction
        if any(word in summary_lower for word in ['error', 'debug', 'fix', 'problem']):
            return "debugging"
        elif any(word in summary_lower for word in ['asset', 'install', 'addon', 'plugin']):
            return "asset_management"
        elif any(word in summary_lower for word in ['scene', 'node', 'script']):
            return "scene_development"
        elif any(word in summary_lower for word in ['animation', 'tween', 'movement']):
            return "animation"
        elif any(word in summary_lower for word in ['physics', 'collision', 'body']):
            return "physics"
        elif any(word in summary_lower for word in ['ui', 'interface', 'button', 'control']):
            return "ui_development"
        else:
            return "general"
    
    async def summarize_conversation_chunk(self, messages: List[Dict], user_id: str = "unknown") -> str:
        """Public API endpoint for frontend to request conversation summarization"""
        if not self.enabled:
            return "[Summarization disabled]"
            
        return await self.create_summary(messages, user_id)
    
    async def update_summary_for_edited_messages(self, messages: List[Dict], user_id: str = "unknown") -> Dict[str, Any]:
        """Update summary when messages are edited"""
        if not self.enabled or not self.weaviate_manager:
            return {"success": False, "error": "Summarization disabled or not available"}
        
        try:
            # Check if we have an existing summary to update
            existing_summary = self.get_summary_with_fuzzy_fallback(user_id, messages)
            
            if existing_summary:
                # Create new summary but mark as updated
                new_summary = await self.create_summary(messages, user_id)
                
                # Store the updated summary
                await self._store_updated_summary(user_id, messages, new_summary, existing_summary)
                
                return {
                    "success": True,
                    "summary": new_summary,
                    "was_updated": True,
                    "previous_summary_found": True,
                    "message": "Summary updated for edited messages"
                }
            else:
                # No existing summary, create new one
                new_summary = await self.create_summary(messages, user_id)
                return {
                    "success": True,
                    "summary": new_summary,
                    "was_updated": False,
                    "previous_summary_found": False,
                    "message": "New summary created"
                }
                
        except Exception as e:
            print(f"MEMORY_UPDATE: Failed to update summary: {e}")
            return {"success": False, "error": str(e)}
    
    async def _store_updated_summary(self, user_id: str, messages: List[Dict], new_summary: str, old_summary: str):
        """Store an updated summary with metadata"""
        if not self.weaviate_manager:
            return
            
        try:
            conversation_hash = self.hash_messages(messages)
            fuzzy_hash = self._create_fuzzy_hash(messages)
            summary_tokens = self.estimate_tokens(new_summary)
            
            # Generate embedding
            if hasattr(self.weaviate_manager, 'openai_client'):
                try:
                    embedding_response = self.weaviate_manager.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=new_summary
                    )
                    summary_embedding = embedding_response.data[0].embedding
                except Exception as e:
                    print(f"MEMORY_EMBEDDING: Failed to generate embedding: {e}")
                    summary_embedding = None
            else:
                summary_embedding = None
            
            # Extract topic
            conversation_topic = self._extract_topic_from_summary(new_summary)
            
            collection = self.weaviate_manager.client.collections.get("ConversationSummary")
            collection.data.insert(
                properties={
                    "user_id": user_id,
                    "conversation_hash": conversation_hash,
                    "fuzzy_hash": fuzzy_hash,
                    "summary": new_summary,
                    "original_message_count": len(messages),
                    "summary_tokens": summary_tokens,
                    "model_used": "updated",  # Special marker for updated summaries
                    "created_at": datetime.now(timezone.utc),
                    "conversation_topic": conversation_topic,
                    "is_updated": True  # Mark as updated
                },
                vector=summary_embedding
            )
            
            print(f"MEMORY_UPDATE: Stored updated summary in Weaviate ({summary_tokens} tokens)")
                
        except Exception as e:
            print(f"MEMORY_UPDATE: Failed to store updated summary: {e}")
    
    def search_similar_conversations(self, query: str, user_id: str, limit: int = 5) -> List[Dict]:
        """Search for similar conversation summaries using semantic search"""
        if not self.weaviate_manager:
            return []
            
        try:
            from weaviate.classes.query import Filter
            
            collection = self.weaviate_manager.client.collections.get("ConversationSummary")
            
            # Generate query embedding
            if hasattr(self.weaviate_manager, 'openai_client'):
                try:
                    embedding_response = self.weaviate_manager.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=query
                    )
                    query_embedding = embedding_response.data[0].embedding
                except Exception:
                    return []  # Fallback to keyword search
            else:
                return []
            
            # Semantic search
            results = collection.query.near_vector(
                near_vector=query_embedding,
                where=Filter.by_property("user_id").equal(user_id),
                limit=limit,
                return_metadata=["distance"]
            )
            
            similar_conversations = []
            for obj in results.objects:
                similar_conversations.append({
                    "summary": obj.properties["summary"],
                    "topic": obj.properties.get("conversation_topic", "general"),
                    "message_count": obj.properties.get("original_message_count", 0),
                    "created_at": obj.properties.get("created_at"),
                    "similarity": 1.0 - obj.metadata.distance if obj.metadata.distance else 0.0
                })
            
            return similar_conversations
            
        except Exception as e:
            print(f"MEMORY_SEARCH: Failed to search similar conversations: {e}")
            return []

# Global instance - will be initialized with weaviate_manager in app.py
conversation_memory = None
