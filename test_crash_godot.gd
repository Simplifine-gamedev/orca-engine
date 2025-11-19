extends Node

# CRASH TEST SCRIPT
# This script will intentionally crash Godot to test crash reporting
# Run this scene to trigger a crash and verify reports are sent to backend

func _ready():
	print("🚨 CRASH TEST: This will crash Godot in 2 seconds...")
	print("🚨 Crash trace will be saved to: project_dir/crashes/crash_TIMESTAMP.txt")
	await get_tree().create_timer(2.0).timeout
	
	print("🚨 TRIGGERING CRASH NOW...")
	
	# Method 1: OS.crash() - simplest and cleanest way
	OS.crash("Testing crash handler - crash trace should be saved to project/crashes/")
	
	# Alternative methods (uncomment to use):
	# trigger_null_crash()  # Method 2: Null pointer dereference
	# trigger_stack_overflow()  # Method 3: Stack overflow

func trigger_null_crash():
	# Access a freed object - this will SIGSEGV
	var node = Node.new()
	node.free()
	node.get_name()  # Crash: accessing freed memory
	
func trigger_stack_overflow():
	# Infinite recursion - stack overflow crash
	trigger_stack_overflow()

func trigger_assertion_crash():
	# Force an assertion failure
	assert(false, "Intentional crash for testing")


