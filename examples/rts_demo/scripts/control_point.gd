extends Node3D
class_name ControlPoint

## Watchtower/Control Point with vision indicator
## Shows eye icon and vision radius to indicate its purpose

@export var vision_radius: float = 20.0
@export var team: int = 0  # 0 = neutral, 1 = player, 2 = enemy
@export var conquered: bool = false

var eye_icon: Sprite3D
var vision_radius_indicator: MeshInstance3D
var tooltip_label: Label3D
var mouse_over: bool = false

# Colors for different team states
const COLOR_NEUTRAL = Color(0.7, 0.7, 0.7, 0.8)
const COLOR_PLAYER = Color(0.2, 0.7, 1.0, 0.8)
const COLOR_ENEMY = Color(1.0, 0.3, 0.2, 0.8)

func _ready():
	_setup_eye_icon()
	_setup_vision_indicator()
	_setup_tooltip()
	_update_visuals()

func _setup_eye_icon():
	"""Create eye icon above the watchtower"""
	eye_icon = Sprite3D.new()
	eye_icon.name = "EyeIcon"
	eye_icon.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	eye_icon.texture = _create_eye_texture()
	eye_icon.pixel_size = 0.01
	eye_icon.position = Vector3(0, 3, 0)  # Float above watchtower
	eye_icon.modulate = COLOR_NEUTRAL
	
	# Add subtle floating animation
	var tween = create_tween().set_loops()
	tween.tween_property(eye_icon, "position:y", 3.3, 1.5).set_ease(Tween.EASE_IN_OUT).set_trans(Tween.TRANS_SINE)
	tween.tween_property(eye_icon, "position:y", 3.0, 1.5).set_ease(Tween.EASE_IN_OUT).set_trans(Tween.TRANS_SINE)
	
	add_child(eye_icon)

func _setup_vision_indicator():
	"""Create vision radius indicator (shown on hover)"""
	vision_radius_indicator = MeshInstance3D.new()
	vision_radius_indicator.name = "VisionRadiusIndicator"
	
	# Create a circle mesh for the vision radius
	var mesh = CylinderMesh.new()
	mesh.top_radius = vision_radius
	mesh.bottom_radius = vision_radius
	mesh.height = 0.1
	mesh.radial_segments = 64
	
	vision_radius_indicator.mesh = mesh
	vision_radius_indicator.position = Vector3(0, 0.05, 0)
	
	# Create semi-transparent material
	var material = StandardMaterial3D.new()
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.albedo_color = Color(1, 1, 1, 0.3)
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	vision_radius_indicator.material_override = material
	
	# Hidden by default, shown on hover
	vision_radius_indicator.visible = false
	add_child(vision_radius_indicator)

func _setup_tooltip():
	"""Create tooltip label"""
	tooltip_label = Label3D.new()
	tooltip_label.name = "Tooltip"
	tooltip_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	tooltip_label.position = Vector3(0, 4, 0)
	tooltip_label.pixel_size = 0.005
	tooltip_label.outline_size = 8
	tooltip_label.font_size = 32
	tooltip_label.modulate = Color(1, 1, 1, 0.9)
	tooltip_label.visible = false
	add_child(tooltip_label)

func _create_eye_texture() -> Texture2D:
	"""Create a simple eye icon texture programmatically"""
	var size = 128
	var img = Image.create(size, size, false, Image.FORMAT_RGBA8)
	
	# Fill with transparent background
	img.fill(Color(0, 0, 0, 0))
	
	# Draw eye shape
	for y in range(size):
		for x in range(size):
			var dx = x - size / 2
			var dy = y - size / 2
			var dist = sqrt(dx * dx + dy * dy)
			
			# Outer eye (ellipse)
			var ellipse_x = dx / (size * 0.45)
			var ellipse_y = dy / (size * 0.3)
			if ellipse_x * ellipse_x + ellipse_y * ellipse_y < 1:
				img.set_pixel(x, y, Color.WHITE)
			
			# Pupil (circle)
			if dist < size * 0.15:
				img.set_pixel(x, y, Color.BLACK)
			# Iris
			elif dist < size * 0.25:
				img.set_pixel(x, y, Color(0.3, 0.5, 0.8, 1.0))
	
	return ImageTexture.create_from_image(img)

func _update_visuals():
	"""Update colors based on team ownership"""
	if not eye_icon:
		return
		
	match team:
		0:  # Neutral
			eye_icon.modulate = COLOR_NEUTRAL
		1:  # Player
			eye_icon.modulate = COLOR_PLAYER
		2:  # Enemy
			eye_icon.modulate = COLOR_ENEMY

func _on_mouse_entered():
	"""Show vision radius and tooltip on hover"""
	mouse_over = true
	if vision_radius_indicator:
		vision_radius_indicator.visible = true
		
		# Update indicator color based on team
		var material = vision_radius_indicator.material_override as StandardMaterial3D
		if material:
			match team:
				0:
					material.albedo_color = Color(COLOR_NEUTRAL.r, COLOR_NEUTRAL.g, COLOR_NEUTRAL.b, 0.3)
				1:
					material.albedo_color = Color(COLOR_PLAYER.r, COLOR_PLAYER.g, COLOR_PLAYER.b, 0.3)
				2:
					material.albedo_color = Color(COLOR_ENEMY.r, COLOR_ENEMY.g, COLOR_ENEMY.b, 0.3)
	
	if tooltip_label:
		tooltip_label.visible = true
		_update_tooltip_text()

func _on_mouse_exited():
	"""Hide vision radius and tooltip"""
	mouse_over = false
	if vision_radius_indicator:
		vision_radius_indicator.visible = false
	if tooltip_label:
		tooltip_label.visible = false

func _update_tooltip_text():
	"""Update tooltip text based on state"""
	if not tooltip_label:
		return
		
	var status = "Neutral" if team == 0 else ("Controlled" if team == 1 else "Enemy")
	tooltip_label.text = "Watchtower [%s]\nVision Radius: %.1fm\n\nConquer to reveal map area" % [status, vision_radius]

func conquer(new_team: int):
	"""Conquer the watchtower for a team"""
	team = new_team
	conquered = (team != 0)
	_update_visuals()
	if mouse_over:
		_update_tooltip_text()
	
	# Play conquest effect
	_play_conquest_effect()

func _play_conquest_effect():
	"""Visual effect when conquered"""
	if not eye_icon:
		return
		
	var tween = create_tween()
	tween.tween_property(eye_icon, "scale", Vector3(1.5, 1.5, 1.5), 0.3)
	tween.tween_property(eye_icon, "scale", Vector3.ONE, 0.3)

# For 3D mouse detection
func _input_event(_camera, event, _position, _normal, _shape_idx):
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			_on_clicked()

func _on_clicked():
	"""Handle watchtower click"""
	print("Watchtower clicked - Team: ", team, " Vision Radius: ", vision_radius)
	# In a real game, this would trigger conquest logic
