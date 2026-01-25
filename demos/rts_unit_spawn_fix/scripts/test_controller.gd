extends Node3D

## Test controller for demonstrating the unit spawn fix
## Press keys to test different spawn scenarios

@onready var building: RTSBuilding = $RTSBuilding if has_node("RTSBuilding") else null

func _ready() -> void:
	print("=== RTS Unit Spawn Test Controller ===")
	print("Press SPACE to spawn a single unit")
	print("Press Q to queue 5 units")
	print("Press W to spawn unit the WRONG WAY (may T-pose)")
	print("Press ESC to quit")
	print("=====================================")
	
	if not building:
		push_error("No building found in scene!")


func _input(event: InputEvent) -> void:
	if not building:
		return
	
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_SPACE:
				print("\n[TEST] Spawning single unit (correct way)")
				building.queue_unit()
			
			KEY_Q:
				print("\n[TEST] Queueing 5 units")
				building.spawn_multiple_units(5)
			
			KEY_W:
				print("\n[TEST] Spawning unit the WRONG WAY (educational)")
				building.spawn_unit_wrong_way()
			
			KEY_ESCAPE:
				print("\nExiting test...")
				get_tree().quit()


func _process(_delta: float) -> void:
	# Display UI hint
	pass
