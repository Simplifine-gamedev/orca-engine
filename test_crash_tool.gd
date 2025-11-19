@tool
extends EditorScript

# CRASH TEST TOOL SCRIPT
# Run this from Script > Run Script to test crash handler
# This will crash the editor and save crash trace to project/crashes/

func _run():
	print("🚨 CRASH TEST: About to crash NOW...")
	print("🚨 Crash trace will be saved to: project_dir/crashes/crash_TIMESTAMP.txt")
	
	# Use OS.crash() - the simplest way to trigger a crash
	OS.crash("Testing crash handler - crash trace should be saved to project/crashes/")

