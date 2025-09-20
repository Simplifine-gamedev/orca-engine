-- Simple LiteLLM Logs Table for Supabase
-- This works with Supabase SQL Editor

-- Create the main logging table
CREATE TABLE IF NOT EXISTS llm_logs (
    -- Core identifiers
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    request_id TEXT NOT NULL,
    
    -- Event type (success, failure, pre_call, post_call, etc.)
    event_type TEXT NOT NULL DEFAULT 'unknown',
    
    -- Model and provider info
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_version TEXT,
    
    -- Request details
    messages_count INTEGER DEFAULT 0,
    input_chars INTEGER DEFAULT 0,
    max_tokens INTEGER,
    temperature DECIMAL(3,2),
    
    -- Token usage
    tokens_prompt INTEGER DEFAULT 0,
    tokens_completion INTEGER DEFAULT 0,
    tokens_total INTEGER DEFAULT 0,
    
    -- Performance metrics
    duration_ms INTEGER DEFAULT 0,
    cost_usd DECIMAL(12,8) DEFAULT 0.0,
    
    -- Success/failure tracking
    success BOOLEAN DEFAULT true NOT NULL,
    error_message TEXT,
    error_type TEXT,
    status_code INTEGER,
    retry_count INTEGER DEFAULT 0,
    
    -- User and project context
    user_id TEXT,
    user_provider TEXT,
    project_id TEXT,
    project_name TEXT,
    session_id TEXT,
    -- ip_address INET, -- Removed for privacy
    user_agent TEXT,
    
    -- Feature usage
    stream BOOLEAN DEFAULT false,
    cache_hit BOOLEAN DEFAULT false,
    thinking_mode BOOLEAN DEFAULT false,
    has_images BOOLEAN DEFAULT false,
    has_tools BOOLEAN DEFAULT false,
    
    -- Tool and feature details
    tools_used JSONB DEFAULT '[]'::jsonb,
    tool_execution_count INTEGER DEFAULT 0,
    tool_execution_time_ms INTEGER DEFAULT 0,
    
    -- Content analysis
    content_type TEXT,
    language_detected TEXT,
    
    -- Conversation and message tracking (for debugging)
    conversation_id TEXT, -- Thread ID to group related messages
    message_type TEXT, -- 'user_input', 'assistant_output', 'tool_call', 'tool_result', 'system'
    message_content TEXT, -- Actual message content (truncated for storage)
    message_content_full JSONB, -- Full message structure for complex content
    message_index INTEGER DEFAULT 0, -- Position in conversation
    parent_message_id TEXT, -- For tracking tool call → tool result relationships
    
    -- System context
    godot_version TEXT,
    backend_version TEXT,
    deployment_mode TEXT DEFAULT 'unknown',
    
    -- Geographical context
    country_code TEXT,
    region TEXT,
    
    -- Business metrics
    is_billable BOOLEAN DEFAULT true,
    billing_category TEXT DEFAULT 'standard',
    
    -- Data quality
    data_quality_score DECIMAL(3,2) DEFAULT 1.0,
    has_pii BOOLEAN DEFAULT false,
    
    -- Raw data for debugging (optional)
    raw_request JSONB,
    raw_response JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_llm_logs_created_at ON llm_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_logs_request_id ON llm_logs(request_id);
CREATE INDEX IF NOT EXISTS idx_llm_logs_user_id ON llm_logs(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_logs_project_id ON llm_logs(project_id) WHERE project_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_logs_model ON llm_logs(model);
CREATE INDEX IF NOT EXISTS idx_llm_logs_provider ON llm_logs(provider);
CREATE INDEX IF NOT EXISTS idx_llm_logs_success ON llm_logs(success);
CREATE INDEX IF NOT EXISTS idx_llm_logs_event_type ON llm_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_llm_logs_conversation_id ON llm_logs(conversation_id) WHERE conversation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_logs_message_type ON llm_logs(message_type) WHERE message_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_logs_message_index ON llm_logs(conversation_id, message_index) WHERE conversation_id IS NOT NULL;

-- Enable Row Level Security
ALTER TABLE llm_logs ENABLE ROW LEVEL SECURITY;

-- Create policy for service role access
CREATE POLICY "Service role full access" ON llm_logs
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

-- Create policy for authenticated users to view their own logs
CREATE POLICY "Authenticated users can view their own logs" ON llm_logs
    FOR SELECT TO authenticated
    USING (auth.uid()::text = user_id);

-- Simple analytics view for hourly usage
CREATE OR REPLACE VIEW llm_usage_hourly AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    model,
    provider,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE success = true) as successful_requests,
    COUNT(*) FILTER (WHERE success = false) as failed_requests,
    ROUND(AVG(duration_ms), 2) as avg_duration_ms,
    ROUND(SUM(cost_usd), 4) as total_cost_usd,
    SUM(tokens_total) as total_tokens,
    ROUND(AVG(tokens_total), 0) as avg_tokens_per_request,
    COUNT(DISTINCT user_id) as unique_users
FROM llm_logs
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1, 2, 3
ORDER BY 1 DESC, 4 DESC;

-- User analytics view 
CREATE OR REPLACE VIEW llm_user_analytics AS
SELECT 
    user_id,
    user_provider,
    COUNT(*) as total_requests,
    COUNT(DISTINCT DATE(created_at)) as active_days,
    ROUND(SUM(cost_usd), 4) as total_cost_usd,
    SUM(tokens_total) as total_tokens,
    MAX(created_at) as last_activity,
    ROUND(AVG(duration_ms), 2) as avg_response_time_ms
FROM llm_logs
WHERE user_id IS NOT NULL
GROUP BY 1, 2
ORDER BY total_requests DESC;

-- Error analysis view
CREATE OR REPLACE VIEW llm_error_analysis AS
SELECT 
    DATE_TRUNC('day', created_at) as day,
    error_type,
    model,
    provider,
    COUNT(*) as error_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY DATE_TRUNC('day', created_at)), 2) as error_percentage
FROM llm_logs
WHERE success = false AND created_at >= NOW() - INTERVAL '30 days'
GROUP BY 1, 2, 3, 4
ORDER BY 1 DESC, 5 DESC;

-- Function for data cleanup
CREATE OR REPLACE FUNCTION cleanup_old_llm_logs(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM llm_logs 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add table comments
COMMENT ON TABLE llm_logs IS 'Comprehensive logging table for all LiteLLM API calls';
COMMENT ON COLUMN llm_logs.request_id IS 'Unique identifier for tracking requests across multiple events';
COMMENT ON COLUMN llm_logs.tools_used IS 'JSON array of tool names used in the request';
COMMENT ON COLUMN llm_logs.cost_usd IS 'Calculated cost in USD with high precision';

-- Create debugging views for conversation analysis
CREATE OR REPLACE VIEW conversation_flow AS
SELECT 
    conversation_id,
    message_index,
    created_at,
    message_type,
    user_id,
    request_id,
    model,
    LEFT(message_content, 100) as content_preview,
    duration_ms,
    tokens_total,
    cost_usd,
    success,
    error_message
FROM llm_logs
WHERE conversation_id IS NOT NULL
ORDER BY conversation_id, message_index;

-- View for debugging specific conversation threads
CREATE OR REPLACE VIEW conversation_debug AS
SELECT 
    conversation_id,
    COUNT(*) as total_messages,
    COUNT(*) FILTER (WHERE message_type = 'user_input') as user_messages,
    COUNT(*) FILTER (WHERE message_type = 'assistant_output') as assistant_messages,
    COUNT(*) FILTER (WHERE message_type = 'tool_call') as tool_calls,
    COUNT(*) FILTER (WHERE message_type = 'tool_result') as tool_results,
    COUNT(*) FILTER (WHERE success = false) as error_count,
    SUM(cost_usd) as total_cost,
    SUM(tokens_total) as total_tokens,
    MIN(created_at) as conversation_start,
    MAX(created_at) as conversation_end,
    MAX(created_at) - MIN(created_at) as conversation_duration,
    ARRAY_AGG(DISTINCT model) as models_used,
    string_agg(DISTINCT unnest(string_to_array(
        array_to_string(
            ARRAY(SELECT jsonb_array_elements_text(tools_used)), ','
        ), ','
    )), ', ') as tools_used_flat
FROM llm_logs
WHERE conversation_id IS NOT NULL
GROUP BY conversation_id
ORDER BY conversation_start DESC;

-- View for message type filtering (great for debugging)
CREATE OR REPLACE VIEW message_types_summary AS
SELECT 
    message_type,
    COUNT(*) as message_count,
    COUNT(DISTINCT conversation_id) as conversations_affected,
    AVG(duration_ms) as avg_duration_ms,
    SUM(cost_usd) as total_cost,
    COUNT(*) FILTER (WHERE success = false) as error_count
FROM llm_logs
WHERE message_type IS NOT NULL
GROUP BY message_type
ORDER BY message_count DESC;

-- Function to get a complete conversation (for debugging)
CREATE OR REPLACE FUNCTION get_conversation_history(conv_id TEXT)
RETURNS TABLE(
    message_index INTEGER,
    created_at TIMESTAMPTZ,
    message_type TEXT,
    content_preview TEXT,
    full_content JSONB,
    model TEXT,
    tokens INTEGER,
    duration_ms INTEGER,
    success BOOLEAN,
    error_message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        l.message_index,
        l.created_at,
        l.message_type,
        LEFT(l.message_content, 200) as content_preview,
        l.message_content_full,
        l.model,
        l.tokens_total,
        l.duration_ms,
        l.success,
        l.error_message
    FROM llm_logs l
    WHERE l.conversation_id = conv_id
    ORDER BY l.message_index;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add helpful comments
COMMENT ON COLUMN llm_logs.conversation_id IS 'Groups messages belonging to the same conversation thread';
COMMENT ON COLUMN llm_logs.message_type IS 'Type of message: user_input, assistant_output, tool_call, tool_result, system';
COMMENT ON COLUMN llm_logs.message_content IS 'Truncated message content for quick debugging (max ~2000 chars)';
COMMENT ON COLUMN llm_logs.message_content_full IS 'Complete message structure including metadata';
COMMENT ON COLUMN llm_logs.message_index IS 'Sequential position in conversation for proper ordering';

-- Success message
SELECT 'LiteLLM logging table with conversation tracking created successfully! ✅' as result;
