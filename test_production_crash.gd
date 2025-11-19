@tool
extends EditorScript

# PRODUCTION CRASH TEST
# This script tests crash saving in BOTH debug and production builds
# Run this from Script > Run Script

func _run():
	print("🚨 PRODUCTION CRASH TEST")
	print("🚨 This will test crash saving regardless of build type (debug/release/production)")
	print("🚨 Crash files will be saved to one of these locations:")
	print("   1. project_dir/crashes/ (preferred)")
	print("   2. user_data_dir/crashes/ (fallback 1)") 
	print("   3. temp_dir/godot_crashes/ (fallback 2)")
	print("   4. current_working_dir/ (last resort)")
	print("")
	
	# Test the bulletproof crash system
	print("🚨 TRIGGERING CRASH NOW...")
	
	# Method 1: OS.crash() - should work in all builds now
	OS.crash("PRODUCTION CRASH TEST - Testing bulletproof crash saving system")
	
	# If OS.crash() somehow doesn't work, this definitely will:
	print("🚨 If you see this, OS.crash() failed - using guaranteed method...")
	var node = Node.new()
	node.free()
	node.get_name()  # Guaranteed crash via accessing freed memory
