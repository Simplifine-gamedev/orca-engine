extends Camera3D

## Simple RTS-style camera controller
## WASD to move, Mouse wheel to zoom, Middle mouse to rotate

@export var move_speed: float = 20.0
@export var zoom_speed: float = 2.0
@export var rotation_speed: float = 0.5
@export var min_height: float = 5.0
@export var max_height: float = 50.0

var camera_height: float = 20.0
var rotating: bool = false
var last_mouse_pos: Vector2

func _ready():
	position = Vector3(0, camera_height, camera_height)
	look_at(Vector3.ZERO, Vector3.UP)

func _process(delta):
	_handle_movement(delta)

func _input(event):
	# Zoom with mouse wheel
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_height = clamp(camera_height - zoom_speed, min_height, max_height)
			_update_camera_position()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_height = clamp(camera_height + zoom_speed, min_height, max_height)
			_update_camera_position()
		elif event.button_index == MOUSE_BUTTON_MIDDLE:
			rotating = event.pressed
			if rotating:
				last_mouse_pos = event.position
	
	# Rotate with middle mouse drag
	if event is InputEventMouseMotion and rotating:
		var delta_mouse = event.position - last_mouse_pos
		rotate_y(-delta_mouse.x * rotation_speed * 0.01)
		last_mouse_pos = event.position

func _handle_movement(delta):
	var direction = Vector3.ZERO
	
	# Get movement input
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		direction -= transform.basis.z
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
		direction += transform.basis.z
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		direction -= transform.basis.x
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		direction += transform.basis.x
	
	direction.y = 0
	direction = direction.normalized()
	
	if direction != Vector3.ZERO:
		position += direction * move_speed * delta

func _update_camera_position():
	var distance = position.distance_to(Vector3.ZERO)
	var direction = (position - Vector3.ZERO).normalized()
	position = Vector3.ZERO + direction * camera_height
