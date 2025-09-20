-- Add conversation tracking columns to existing llm_logs table
-- Run this in your Supabase SQL Editor to enhance your existing table

-- Add conversation and message tracking columns
ALTER TABLE llm_logs ADD COLUMN IF NOT EXISTS conversation_id TEXT;
ALTER TABLE llm_logs ADD COLUMN IF NOT EXISTS message_type TEXT;
ALTER TABLE llm_logs ADD COLUMN IF NOT EXISTS message_content TEXT;
ALTER TABLE llm_logs ADD COLUMN IF NOT EXISTS message_content_full JSONB;
ALTER TABLE llm_logs ADD COLUMN IF NOT EXISTS message_index INTEGER DEFAULT 0;
ALTER TABLE llm_logs ADD COLUMN IF NOT EXISTS parent_message_id TEXT;

-- Update event_type to support assistant_response
-- (Only run this if you have constraints on event_type)
-- ALTER TABLE llm_logs DROP CONSTRAINT IF EXISTS llm_logs_event_type_check;
-- ALTER TABLE llm_logs ADD CONSTRAINT llm_logs_event_type_check 
--   CHECK (event_type IN ('pre_call', 'post_call', 'success', 'failure', 'assistant_response', 'timeout', 'rate_limit'));

-- Add indexes for the new columns
CREATE INDEX IF NOT EXISTS idx_llm_logs_conversation_id ON llm_logs(conversation_id) WHERE conversation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_llm_logs_message_type ON llm_logs(message_type) WHERE message_type IS NOT NULL;

-- Add helpful comments
COMMENT ON COLUMN llm_logs.conversation_id IS 'Groups messages in the same conversation thread';
COMMENT ON COLUMN llm_logs.message_type IS 'user_input, assistant_output, tool_call, tool_result, system';
COMMENT ON COLUMN llm_logs.message_content IS 'Truncated message content (max 2000 chars)';
COMMENT ON COLUMN llm_logs.message_content_full IS 'Complete message structure as JSON';

-- Create simple debugging view
CREATE OR REPLACE VIEW recent_conversations AS
SELECT 
    conversation_id,
    message_type,
    LEFT(message_content, 100) as content_preview,
    model,
    created_at,
    duration_ms,
    tokens_total,
    cost_usd,
    success
FROM llm_logs 
WHERE conversation_id IS NOT NULL 
  AND created_at >= NOW() - INTERVAL '6 hours'
ORDER BY conversation_id, message_index;
