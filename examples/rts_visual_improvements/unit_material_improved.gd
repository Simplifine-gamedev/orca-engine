extends MeshInstance3D
## Improved unit material for better visibility in RTS games
## Addresses ORC-142: Improve unit colors/contrast

@export_group("Unit Appearance")
@export var team_color := Color(0.8, 0.2, 0.2)  # Red team
@export_range(0.0, 1.0, 0.05) var color_brightness := 1.2
@export var use_emission := true
@export_range(0.0, 1.0, 0.1) var emission_strength := 0.3

@export_group("Outline")
@export var add_outline := true
@export var outline_color := Color(1.0, 1.0, 1.0, 0.8)
@export_range(0.0, 0.1, 0.001) var outline_width := 0.02

@export_group("Selection")
@export var is_selected := false
@export var selection_color := Color(1.0, 1.0, 0.0)
@export var selection_pulse := true

var material: StandardMaterial3D
var outline_mesh: MeshInstance3D
var time := 0.0

func _ready():
	setup_unit_material()
	if add_outline:
		create_outline()

func _process(delta):
	if is_selected and selection_pulse:
		time += delta
		var pulse = (sin(time * 5.0) + 1.0) / 2.0  # 0 to 1
		update_selection_pulse(pulse)

func setup_unit_material():
	"""Create material with high visibility"""
	material = StandardMaterial3D.new()
	
	# Base color (brightened for visibility)
	var visible_color = team_color * color_brightness
	material.albedo_color = visible_color
	
	# Material properties for good visibility
	material.roughness = 0.6
	material.metallic = 0.3
	material.specular_mode = BaseMaterial3D.SPECULAR_SCHLICK_GGX
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	
	# Emission for better visibility
	if use_emission:
		material.emission_enabled = true
		material.emission = visible_color
		material.emission_energy_multiplier = emission_strength
	
	# Rim lighting for edge definition
	material.rim_enabled = true
	material.rim = 0.5
	material.rim_tint = 0.3
	
	# Shadows
	material.disable_receive_shadows = false
	
	# Apply material
	self.material_override = material

func create_outline():
	"""Create outline effect for better unit definition"""
	if not mesh:
		return
	
	outline_mesh = MeshInstance3D.new()
	outline_mesh.mesh = mesh
	
	# Outline material (inverted normals, slightly larger)
	var outline_mat = StandardMaterial3D.new()
	outline_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	outline_mat.albedo_color = outline_color
	outline_mat.cull_mode = BaseMaterial3D.CULL_FRONT  # Show only backfaces
	outline_mat.disable_ambient_light = true
	outline_mat.disable_fog = true
	
	outline_mesh.material_override = outline_mat
	outline_mesh.scale = Vector3(1.0 + outline_width, 1.0 + outline_width, 1.0 + outline_width)
	
	add_child(outline_mesh)

func set_selected(selected: bool):
	"""Update unit appearance for selection state"""
	is_selected = selected
	
	if selected:
		if material:
			material.emission_enabled = true
			material.emission = selection_color
			material.emission_energy_multiplier = 0.5
		
		if outline_mesh and outline_mesh.material_override:
			outline_mesh.material_override.albedo_color = selection_color
	else:
		setup_unit_material()  # Reset to normal appearance

func update_selection_pulse(pulse_value: float):
	"""Animate selection indication"""
	if not material or not is_selected:
		return
	
	var energy = lerp(0.3, 0.7, pulse_value)
	material.emission_energy_multiplier = energy

func set_team_color(color: Color):
	"""Change unit team color"""
	team_color = color
	setup_unit_material()

# Predefined team colors (highly visible)
static func get_team_colors() -> Array[Color]:
	return [
		Color(0.9, 0.2, 0.2),   # Red
		Color(0.2, 0.4, 0.9),   # Blue
		Color(0.2, 0.8, 0.3),   # Green
		Color(0.9, 0.8, 0.2),   # Yellow
		Color(0.8, 0.2, 0.8),   # Purple
		Color(0.2, 0.8, 0.8),   # Cyan
		Color(0.9, 0.5, 0.2),   # Orange
		Color(0.9, 0.9, 0.9)    # White
	]
