extends MeshInstance3D
## Improved terrain rendering for RTS games
## Addresses ORC-142: Better ground texture and visual variety

@export_group("Terrain Appearance")
@export var base_color := Color(0.45, 0.55, 0.35)  # Grass green
@export var secondary_color := Color(0.35, 0.4, 0.3)  # Darker variation
@export_range(0.0, 1.0, 0.05) var roughness := 0.8
@export_range(0.0, 1.0, 0.05) var metallic := 0.0

@export_group("Visual Features")
@export var use_triplanar_mapping := true
@export var add_grid_overlay := false  # Helpful for RTS gameplay
@export var grid_color := Color(0.3, 0.3, 0.3, 0.2)
@export_range(1.0, 20.0, 0.5) var grid_size := 5.0

@export_group("Height Variation")
@export var height_color_variation := true
@export var low_altitude_color := Color(0.4, 0.5, 0.35)
@export var high_altitude_color := Color(0.6, 0.65, 0.55)

var material: StandardMaterial3D

func _ready():
	setup_terrain_material()

func setup_terrain_material():
	"""Create material with better visibility and contrast"""
	material = StandardMaterial3D.new()
	
	# Base appearance
	material.albedo_color = base_color
	material.roughness = roughness
	material.metallic = metallic
	
	# Improve visibility
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.specular_mode = BaseMaterial3D.SPECULAR_SCHLICK_GGX
	
	# Disable unnecessary features that can darken terrain
	material.ao_enabled = false
	
	# Add subtle normal variation for visual interest
	material.normal_enabled = false  # Can enable with texture
	
	# Enable shadows receiving
	material.disable_receive_shadows = false
	
	# Slight emission for better visibility
	material.emission_enabled = true
	material.emission = base_color * 0.1  # Very subtle glow
	material.emission_energy_multiplier = 0.2
	
	# Triplanar for better texturing on slopes
	if use_triplanar_mapping:
		material.uv1_triplanar = true
		material.uv1_world_triplanar = true
	
	# Apply material
	self.material_override = material
	
	# Add grid overlay if enabled
	if add_grid_overlay:
		add_grid_material()

func add_grid_material():
	"""Add grid overlay for RTS-style gameplay clarity"""
	var shader_material = ShaderMaterial.new()
	var shader = Shader.new()
	
	shader.code = """
shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back;

uniform vec4 grid_color : source_color = vec4(0.3, 0.3, 0.3, 0.2);
uniform float grid_size = 5.0;
uniform float line_width = 0.05;

void fragment() {
	vec3 world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec2 grid_pos = world_pos.xz / grid_size;
	vec2 grid_fract = fract(grid_pos);
	
	// Create grid lines
	float grid_line_x = step(1.0 - line_width, grid_fract.x) + step(grid_fract.x, line_width);
	float grid_line_y = step(1.0 - line_width, grid_fract.y) + step(grid_fract.y, line_width);
	float grid = clamp(grid_line_x + grid_line_y, 0.0, 1.0);
	
	ALBEDO = mix(ALBEDO, grid_color.rgb, grid * grid_color.a);
}
"""
	
	shader_material.shader = shader
	shader_material.set_shader_parameter("grid_color", grid_color)
	shader_material.set_shader_parameter("grid_size", grid_size)
	self.material_overlay = shader_material

func create_terrain_mesh(width: int, depth: int, cell_size: float = 1.0) -> ArrayMesh:
	"""Generate terrain mesh with height variation"""
	var surface_array = []
	surface_array.resize(Mesh.ARRAY_MAX)
	
	var vertices = PackedVector3Array()
	var normals = PackedVector3Array()
	var uvs = PackedVector2Array()
	var indices = PackedInt32Array()
	
	var noise = FastNoiseLite.new()
	noise.seed = randi()
	noise.frequency = 0.02
	noise.fractal_octaves = 3
	
	# Generate vertices
	for z in range(depth + 1):
		for x in range(width + 1):
			var pos_x = (x - width / 2.0) * cell_size
			var pos_z = (z - depth / 2.0) * cell_size
			var height = noise.get_noise_2d(x, z) * 3.0  # Height variation
			
			vertices.append(Vector3(pos_x, height, pos_z))
			normals.append(Vector3.UP)  # Will be recalculated
			uvs.append(Vector2(float(x) / width, float(z) / depth))
	
	# Generate indices
	for z in range(depth):
		for x in range(width):
			var i = z * (width + 1) + x
			
			# Two triangles per cell
			indices.append(i)
			indices.append(i + width + 1)
			indices.append(i + 1)
			
			indices.append(i + 1)
			indices.append(i + width + 1)
			indices.append(i + width + 2)
	
	# Calculate proper normals for lighting
	var normals_calculated = PackedVector3Array()
	normals_calculated.resize(vertices.size())
	for i in range(indices.size() / 3):
		var i0 = indices[i * 3]
		var i1 = indices[i * 3 + 1]
		var i2 = indices[i * 3 + 2]
		
		var v0 = vertices[i0]
		var v1 = vertices[i1]
		var v2 = vertices[i2]
		
		var normal = (v1 - v0).cross(v2 - v0).normalized()
		normals_calculated[i0] += normal
		normals_calculated[i1] += normal
		normals_calculated[i2] += normal
	
	for i in range(normals_calculated.size()):
		normals_calculated[i] = normals_calculated[i].normalized()
	
	surface_array[Mesh.ARRAY_VERTEX] = vertices
	surface_array[Mesh.ARRAY_NORMAL] = normals_calculated
	surface_array[Mesh.ARRAY_TEX_UV] = uvs
	surface_array[Mesh.ARRAY_INDEX] = indices
	
	var array_mesh = ArrayMesh.new()
	array_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, surface_array)
	
	return array_mesh

func apply_height_based_coloring():
	"""Vary terrain color based on height for visual interest"""
	if not height_color_variation or not mesh:
		return
	
	# This would require vertex colors or a custom shader
	# For now, we'll use a shader
	var shader_material = ShaderMaterial.new()
	var shader = Shader.new()
	
	shader.code = """
shader_type spatial;

uniform vec4 low_color : source_color = vec4(0.4, 0.5, 0.35, 1.0);
uniform vec4 high_color : source_color = vec4(0.6, 0.65, 0.55, 1.0);
uniform float height_range = 10.0;

void fragment() {
	vec3 world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	float height_factor = clamp(world_pos.y / height_range, 0.0, 1.0);
	ALBEDO = mix(low_color.rgb, high_color.rgb, height_factor);
	ROUGHNESS = 0.8;
	METALLIC = 0.0;
}
"""
	
	shader_material.shader = shader
	shader_material.set_shader_parameter("low_color", low_altitude_color)
	shader_material.set_shader_parameter("high_color", high_altitude_color)
	self.material_override = shader_material
