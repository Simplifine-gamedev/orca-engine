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
        "cerebras/llama3.1-70b",              # Fast, reliable for summarization
        "cerebras/gpt-oss-120b",              # Secondary fast option
        "openai/gpt-4o-mini",                 # Fallback - avoid gpt-5-mini timeout issues
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
    
    # Summarization settings (enhanced for comprehensive summaries)
    SUMMARY_MAX_TOKENS = 800      # Increased for comprehensive technical summaries
    SUMMARY_TEMPERATURE = 0.2     # Lower for more focused, technical summaries
    SUMMARY_TIMEOUT = 45          # Longer timeout for detailed analysis
    
    # Memory management settings (aligned with user requirements)
    KEEP_RECENT_MESSAGES = 20     # Keep last 20 messages visible to model
    MIN_MESSAGES_TO_PRUNE = 5     # Don't prune very short conversations
    SUMMARIZATION_TRIGGER = 0.5   # Trigger at 50% token usage
    CLEANUP_DAYS_DEFAULT = 30     # Default days for cleanup
    
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
        return os.getenv('MEMORY_SUMMARY_PROMPT', """You are an expert at creating comprehensive, technically detailed summaries of Godot game development conversations.

Create a summary that preserves ALL essential context for seamless conversation continuation:

CRITICAL REQUIREMENTS:
1. **Project Structure**: Scene files, key nodes, scripts, and their relationships
2. **Technical Decisions**: Code changes, configurations, settings, resource paths
3. **Problem-Solution Mapping**: Issues encountered and exact solutions implemented
4. **Tool Usage History**: Which tools were used, with what parameters, and outcomes
5. **Code Context**: Functions modified, classes created, variable names, property values
6. **Asset Information**: Images, materials, meshes, shaders created or modified
7. **User Patterns**: Preferences, naming conventions, architectural choices
8. **Current State**: What's working, what's broken, what's in progress

ANALYSIS DEPTH:
- Extract specific file paths, node names, and property values
- Note exact tool parameters that worked vs failed
- Preserve debugging insights and error patterns
- Include resource IDs, shader parameters, animation settings
- Document input mappings, physics settings, scene hierarchy

Conversation to summarize:
{conversation_text}

Create a comprehensive technical summary (400-800 words) that includes:
- Executive summary of what was accomplished
- Detailed technical context for immediate continuation
- Specific names, paths, and values for reference
- Problem-solving insights and patterns discovered

IMPORTANT: This summary will replace the original messages in the AI's context, so it must contain ALL information needed for the AI to continue helping effectively.

Technical Summary:""")

# Environment variable examples for .env file:
"""
# Enable/disable intelligent memory management (works with existing Weaviate setup)
ENABLE_INTELLIGENT_MEMORY=true

# Custom summarization models (comma-separated, in order of preference)
MEMORY_SUMMARIZATION_MODELS=openai/gpt-4o-mini,cerebras/gpt-oss-120b,cerebras/qwen-3-coder-480b

# Weaviate configuration (reuses existing WEAVIATE_URL and WEAVIATE_API_KEY)
# WEAVIATE_URL=https://your-cluster.weaviate.cloud
# WEAVIATE_API_KEY=your-weaviate-api-key

# Custom summary prompt template (optional)
MEMORY_SUMMARY_PROMPT=Custom prompt here...
"""
