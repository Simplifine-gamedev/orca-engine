extends Node3D

## DemoController - Interactive demo controls for testing decoration systems
##
## Press keys to regenerate different systems:
## - R: Regenerate terrain
## - V: Regenerate vegetation
## - D: Regenerate decorations
## - A: Regenerate all

@onready var terrain := $"../HeightmapTerrain" as HeightmapTerrain
@onready var vegetation := $"../VegetationSystem" as VegetationSystem
@onready var decorations := $"../DecorationSpawner" as DecorationSpawner
@onready var camera := $"../Camera3D" as Camera3D

var camera_distance := 40.0
var camera_angle := 45.0
var camera_rotation := 0.0


func _ready() -> void:
	print("=== RTS Demo - Map Visual Decorations ===")
	print("Controls:")
	print("  R - Regenerate terrain")
	print("  V - Regenerate vegetation")
	print("  D - Regenerate decorations")
	print("  A - Regenerate all")
	print("  Arrow Keys - Rotate camera")
	print("  + / - - Zoom camera")
	print("")
	_update_camera()


func _process(delta: float) -> void:
	# Camera rotation
	if Input.is_action_pressed("ui_left"):
		camera_rotation -= 60.0 * delta
		_update_camera()
	
	if Input.is_action_pressed("ui_right"):
		camera_rotation += 60.0 * delta
		_update_camera()
	
	# Camera zoom
	if Input.is_action_pressed("ui_page_up"):
		camera_distance = max(20.0, camera_distance - 30.0 * delta)
		_update_camera()
	
	if Input.is_action_pressed("ui_page_down"):
		camera_distance = min(100.0, camera_distance + 30.0 * delta)
		_update_camera()


func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_R:
				print("Regenerating terrain...")
				if terrain:
					terrain.regenerate()
			
			KEY_V:
				print("Regenerating vegetation...")
				if vegetation:
					vegetation.regenerate()
			
			KEY_D:
				print("Regenerating decorations...")
				if decorations:
					decorations.regenerate()
			
			KEY_A:
				print("Regenerating everything...")
				if terrain:
					terrain.regenerate()
				await get_tree().create_timer(0.1).timeout
				if vegetation:
					vegetation.regenerate()
				if decorations:
					decorations.regenerate()


func _update_camera() -> void:
	if not camera:
		return
	
	var angle_rad := deg_to_rad(camera_angle)
	var rotation_rad := deg_to_rad(camera_rotation)
	
	var x := cos(rotation_rad) * camera_distance * cos(angle_rad)
	var y := camera_distance * sin(angle_rad)
	var z := sin(rotation_rad) * camera_distance * cos(angle_rad)
	
	camera.position = Vector3(x, y, z)
	camera.look_at(Vector3.ZERO, Vector3.UP)
