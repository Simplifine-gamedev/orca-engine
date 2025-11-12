-- Migration: Add frontend_logs column to 3D jobs table
-- Run this in your Supabase SQL editor to enable frontend error logging
-- This assumes the table is named 'jobs' or similar - adjust table name as needed

-- Add frontend_logs column as JSONB to store structured log entries
-- This allows appending logs without overwriting existing data
ALTER TABLE IF EXISTS jobs 
ADD COLUMN IF NOT EXISTS frontend_logs JSONB DEFAULT '[]'::jsonb;

-- Add index for efficient querying of logs
CREATE INDEX IF NOT EXISTS idx_jobs_frontend_logs ON jobs USING GIN (frontend_logs);

-- Add comment for documentation
COMMENT ON COLUMN jobs.frontend_logs IS 'Array of frontend log entries. Each entry contains: timestamp, level (error/warning/info), message, context (optional), stack_trace (optional)';

-- Example structure of log entries:
-- [
--   {
--     "timestamp": "2025-01-15T10:30:45Z",
--     "level": "error",
--     "message": "HTTP request failed",
--     "context": {
--       "url": "/api/jobs/text-to-3d",
--       "status_code": 500,
--       "user_id": "godot_abc123",
--       "job_id": "job_xyz789"
--     },
--     "stack_trace": "optional stack trace if available"
--   }
-- ]

