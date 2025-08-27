"""
© 2025 Simplifine Corp. Original backend contribution for this Godot fork.
Personal Non‑Commercial License applies. Commercial use requires a separate license from Simplifine.
See LICENSES/COMPANY-NONCOMMERCIAL.md.

Configuration for intelligent conversation memory management.
"""
import os
from typing import List

class MemoryConfig:
    """Configuration for conversation memory management"""
    
    # Default summarization models in order of preference
    DEFAULT_SUMMARIZATION_MODELS = [
        "cerebras/gpt-oss-120b",              # Fast, efficient for summarization
        "cerebras/qwen-3-coder-480b",         # Good for code context (Cerebras version)
        "openai/gpt-4o-mini",                 # Fallback - cost effective
    ]
    
    # Storage configuration (cloud-native)
    WEAVIATE_COLLECTION_NAME = "ConversationSummary"
    
    # Token limits with safety margins
    TOKEN_LIMITS = {
        "anthropic/claude-sonnet-4-20250514": 180000,  # 200k - 20k margin
        "openai/gpt-5": 120000,                        # 128k - 8k margin  
        "openai/gpt-4o": 120000,                       # 128k - 8k margin
        "openai/gpt-4-turbo": 120000,                  # 128k - 8k margin
        "anthropic/claude-3-5-sonnet-20241022": 180000, # 200k - 20k margin
        "meta-llama/llama-3.1-70b-instruct": 120000,   # 128k - 8k margin
        "cerebras/qwen-3-coder-480b": 30000,            # 32k - 2k margin (Cerebras Qwen)
        "cerebras/gpt-oss-120b": 7000,                  # 8k - 1k margin
    }
    
    # Summarization settings
    SUMMARY_MAX_TOKENS = 500
    SUMMARY_TEMPERATURE = 0.3
    SUMMARY_TIMEOUT = 30
    
    # Memory management settings
    KEEP_RECENT_MESSAGES = 15    # Number of recent messages to always keep
    MIN_MESSAGES_TO_PRUNE = 3    # Don't prune conversations shorter than this
    CLEANUP_DAYS_DEFAULT = 30    # Default days for cleanup
    
    @classmethod
    def get_summarization_models(cls) -> List[str]:
        """Get summarization models from environment or use defaults"""
        env_models = os.getenv('MEMORY_SUMMARIZATION_MODELS')
        if env_models:
            return [model.strip() for model in env_models.split(',')]
        return cls.DEFAULT_SUMMARIZATION_MODELS
    
    @classmethod
    def get_collection_name(cls) -> str:
        """Get Weaviate collection name for conversation summaries"""
        return os.getenv('WEAVIATE_CONVERSATION_COLLECTION', cls.WEAVIATE_COLLECTION_NAME)
    
    @classmethod
    def get_token_limit(cls, model: str) -> int:
        """Get token limit for a model"""
        return cls.TOKEN_LIMITS.get(model, 100000)  # Conservative default
    
    @classmethod  
    def is_enabled(cls) -> bool:
        """Check if intelligent memory management is enabled"""
        return os.getenv('ENABLE_INTELLIGENT_MEMORY', 'true').lower() == 'true'
    
    @classmethod
    def get_summary_prompt_template(cls) -> str:
        """Get the prompt template for summarization"""
        return os.getenv('MEMORY_SUMMARY_PROMPT', """You are an expert at creating concise, contextually rich summaries of technical conversations about Godot game development.

Analyze this conversation and create a comprehensive summary that preserves:
1. Key technical decisions and solutions discussed
2. Important code changes, file paths, and configurations  
3. Problems identified and how they were resolved
4. Tool usage patterns and outcomes
5. User preferences and project context

Conversation to summarize:
{conversation_text}

Create a detailed but concise summary (aim for 200-400 words) that would help an AI assistant understand the context if this conversation continued later. Focus on actionable information and technical details that would be relevant for future assistance.

Summary:""")

# Environment variable examples for .env file:
"""
# Enable/disable intelligent memory management (works with existing Weaviate setup)
ENABLE_INTELLIGENT_MEMORY=true

# Custom summarization models (comma-separated, in order of preference)
MEMORY_SUMMARIZATION_MODELS=cerebras/gpt-oss-120b,cerebras/qwen-3-coder-480b,openai/gpt-4o-mini

# Weaviate configuration (reuses existing WEAVIATE_URL and WEAVIATE_API_KEY)
# WEAVIATE_URL=https://your-cluster.weaviate.cloud
# WEAVIATE_API_KEY=your-weaviate-api-key

# Custom summary prompt template (optional)
MEMORY_SUMMARY_PROMPT=Custom prompt here...
"""
