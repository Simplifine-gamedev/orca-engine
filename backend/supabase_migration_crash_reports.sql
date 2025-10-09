-- Crash Reports Table for Orca Engine
-- Run this migration in your Supabase SQL editor to enable crash reporting

-- Create the crash_reports table
CREATE TABLE IF NOT EXISTS crash_reports (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Crash metadata
    report_id TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    project_name TEXT,
    
    -- User identification (anonymized)
    user_id TEXT,
    machine_id TEXT,
    
    -- Crash data
    crash_dump TEXT NOT NULL,
    signal_number INTEGER,
    
    -- Additional context
    timestamp_reported BIGINT,
    user_agent TEXT,
    
    -- Indexing for fast queries
    CONSTRAINT valid_platform CHECK (platform IN ('macos', 'windows', 'linux', 'test'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_crash_reports_created_at ON crash_reports(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_crash_reports_platform ON crash_reports(platform);
CREATE INDEX IF NOT EXISTS idx_crash_reports_machine_id ON crash_reports(machine_id);
CREATE INDEX IF NOT EXISTS idx_crash_reports_project ON crash_reports(project_name);
CREATE INDEX IF NOT EXISTS idx_crash_reports_engine_version ON crash_reports(engine_version);

-- Enable Row Level Security
ALTER TABLE crash_reports ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Allow anyone to insert crash reports (anonymous crash reporting)
CREATE POLICY "Anyone can submit crash reports" ON crash_reports
    FOR INSERT WITH CHECK (true);

-- RLS Policy: Only authenticated users can view crash reports
CREATE POLICY "Authenticated users can view crash reports" ON crash_reports
    FOR SELECT USING (auth.role() = 'authenticated');

-- Add helpful comment
COMMENT ON TABLE crash_reports IS 'Stores crash reports from Orca Engine (Godot fork) clients for debugging and analysis';

-- Grant anon access for inserts (required for public crash reporting)
GRANT INSERT ON crash_reports TO anon;
GRANT SELECT ON crash_reports TO authenticated;

