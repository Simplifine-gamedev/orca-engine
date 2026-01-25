extends Node2D

## Demo controller to test the unit spawning system

@onready var building = $Building


func _ready() -> void:
	print("=== RTS Unit Spawning Demo ===")
	print("This demo shows the fix for ORC-118:")
	print("Units now spawn OUTSIDE buildings, not inside them")
	print("")
	print("Controls:")
	print("  - LEFT CLICK on building: Start/stop production")
	print("  - RIGHT CLICK anywhere: Set rally point")
	print("  - SPACE: Clear rally point")
	print("  - ESC: Quit")


func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			# Check if clicked on building
			if is_point_near_building(event.position):
				toggle_production()
		
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			# Set rally point
			building.set_rally_point(event.position)
	
	elif event is InputEventKey and event.pressed:
		if event.keycode == KEY_SPACE:
			building.clear_rally_point()
		elif event.keycode == KEY_ESCAPE:
			get_tree().quit()


func is_point_near_building(point: Vector2) -> bool:
	var distance = point.distance_to(building.global_position)
	return distance < 50.0


func toggle_production() -> void:
	if building.spawn_timer.is_stopped():
		building.start_production()
		print("Production STARTED - units will spawn outside the building")
	else:
		building.stop_production()
		print("Production STOPPED")
