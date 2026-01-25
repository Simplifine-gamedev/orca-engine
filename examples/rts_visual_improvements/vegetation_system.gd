extends Node3D
## Vegetation system for adding visual interest and depth to RTS maps
## Addresses ORC-142: Add environmental objects (vegetation, rocks, trees)

@export_group("Vegetation Density")
@export_range(0.0, 1.0, 0.05) var tree_density := 0.15
@export_range(0.0, 1.0, 0.05) var rock_density := 0.08
@export_range(0.0, 1.0, 0.05) var grass_patch_density := 0.25

@export_group("Placement Area")
@export var area_size := Vector2(100, 100)
@export var min_distance_between_objects := 3.0
@export var avoid_steep_slopes := true
@export_range(0.0, 90.0, 5.0) var max_slope_angle := 30.0

@export_group("Visual Settings")
@export var use_lod := true  # Level of detail for performance
@export var cast_shadows := true
@export var vegetation_brightness := 1.2  # Slightly brighter for visibility

@export_group("References")
@export var terrain: MeshInstance3D

# Vegetation templates
var tree_meshes: Array[Mesh] = []
var rock_meshes: Array[Mesh] = []
var placed_objects: Array[Node3D] = []

func _ready():
	create_simple_meshes()
	if terrain:
		populate_vegetation()

func create_simple_meshes():
	"""Create simple procedural vegetation meshes"""
	# Simple tree
	var tree_mesh = create_simple_tree()
	tree_meshes.append(tree_mesh)
	
	# Simple rocks
	var rock_mesh = create_simple_rock()
	rock_meshes.append(rock_mesh)

func create_simple_tree() -> ArrayMesh:
	"""Create a simple tree mesh (trunk + foliage)"""
	var surface_tool = SurfaceTool.new()
	surface_tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	
	# Trunk (cylinder approximation)
	var trunk_radius = 0.3
	var trunk_height = 3.0
	var segments = 8
	
	for i in range(segments):
		var angle1 = (float(i) / segments) * TAU
		var angle2 = (float(i + 1) / segments) * TAU
		
		var x1 = cos(angle1) * trunk_radius
		var z1 = sin(angle1) * trunk_radius
		var x2 = cos(angle2) * trunk_radius
		var z2 = sin(angle2) * trunk_radius
		
		# Bottom triangle
		surface_tool.set_normal(Vector3(cos(angle1), 0, sin(angle1)))
		surface_tool.set_color(Color(0.4, 0.3, 0.2) * vegetation_brightness)
		surface_tool.add_vertex(Vector3(x1, 0, z1))
		surface_tool.add_vertex(Vector3(x2, 0, z2))
		surface_tool.add_vertex(Vector3(x1, trunk_height, z1))
		
		# Top triangle
		surface_tool.add_vertex(Vector3(x2, 0, z2))
		surface_tool.add_vertex(Vector3(x2, trunk_height, z2))
		surface_tool.add_vertex(Vector3(x1, trunk_height, z1))
	
	# Foliage (cone)
	var foliage_radius = 2.0
	var foliage_height = 4.0
	var foliage_base_y = trunk_height - 0.5
	
	for i in range(segments):
		var angle1 = (float(i) / segments) * TAU
		var angle2 = (float(i + 1) / segments) * TAU
		
		var x1 = cos(angle1) * foliage_radius
		var z1 = sin(angle1) * foliage_radius
		var x2 = cos(angle2) * foliage_radius
		var z2 = sin(angle2) * foliage_radius
		
		var normal = Vector3((x1 + x2) / 2, foliage_radius, (z1 + z2) / 2).normalized()
		
		surface_tool.set_normal(normal)
		surface_tool.set_color(Color(0.3, 0.6, 0.25) * vegetation_brightness)
		surface_tool.add_vertex(Vector3(x1, foliage_base_y, z1))
		surface_tool.add_vertex(Vector3(x2, foliage_base_y, z2))
		surface_tool.add_vertex(Vector3(0, foliage_base_y + foliage_height, 0))
	
	surface_tool.generate_normals()
	return surface_tool.commit()

func create_simple_rock() -> ArrayMesh:
	"""Create a simple rock mesh"""
	var surface_tool = SurfaceTool.new()
	surface_tool.begin(Mesh.PRIMITIVE_TRIANGLES)
	
	# Simple irregular shape (icosahedron-like)
	var vertices = [
		Vector3(0, 1, 0),
		Vector3(0.8, 0.3, 0),
		Vector3(0.5, 0.3, 0.7),
		Vector3(-0.5, 0.3, 0.7),
		Vector3(-0.8, 0.3, 0),
		Vector3(-0.5, 0.3, -0.7),
		Vector3(0.5, 0.3, -0.7),
		Vector3(0, -0.2, 0)
	]
	
	var rock_color = Color(0.5, 0.5, 0.55) * vegetation_brightness
	
	# Top faces
	var top_indices = [
		[0, 1, 2], [0, 2, 3], [0, 3, 4],
		[0, 4, 5], [0, 5, 6], [0, 6, 1]
	]
	
	for face in top_indices:
		var v0 = vertices[face[0]]
		var v1 = vertices[face[1]]
		var v2 = vertices[face[2]]
		var normal = (v1 - v0).cross(v2 - v0).normalized()
		
		surface_tool.set_normal(normal)
		surface_tool.set_color(rock_color)
		surface_tool.add_vertex(v0)
		surface_tool.add_vertex(v1)
		surface_tool.add_vertex(v2)
	
	# Bottom faces
	var bottom_indices = [
		[7, 2, 1], [7, 3, 2], [7, 4, 3],
		[7, 5, 4], [7, 6, 5], [7, 1, 6]
	]
	
	for face in bottom_indices:
		var v0 = vertices[face[0]]
		var v1 = vertices[face[1]]
		var v2 = vertices[face[2]]
		var normal = (v1 - v0).cross(v2 - v0).normalized()
		
		surface_tool.set_normal(normal)
		surface_tool.set_color(rock_color * 0.8)  # Slightly darker bottom
		surface_tool.add_vertex(v0)
		surface_tool.add_vertex(v1)
		surface_tool.add_vertex(v2)
	
	surface_tool.generate_normals()
	return surface_tool.commit()

func populate_vegetation():
	"""Place vegetation objects across the terrain"""
	clear_vegetation()
	
	var rng = RandomNumberGenerator.new()
	rng.seed = hash(name)
	
	# Calculate total objects to place
	var area = area_size.x * area_size.y
	var num_trees = int(area * tree_density / 100.0)
	var num_rocks = int(area * rock_density / 100.0)
	
	# Place trees
	for i in range(num_trees):
		var pos = get_random_position_on_terrain(rng)
		if pos and is_position_valid(pos):
			place_tree(pos, rng)
	
	# Place rocks
	for i in range(num_rocks):
		var pos = get_random_position_on_terrain(rng)
		if pos and is_position_valid(pos):
			place_rock(pos, rng)

func get_random_position_on_terrain(rng: RandomNumberGenerator) -> Vector3:
	"""Get random position on terrain surface"""
	if not terrain:
		var x = rng.randf_range(-area_size.x / 2, area_size.x / 2)
		var z = rng.randf_range(-area_size.y / 2, area_size.y / 2)
		return Vector3(x, 0, z)
	
	# Sample terrain height
	var x = rng.randf_range(-area_size.x / 2, area_size.x / 2)
	var z = rng.randf_range(-area_size.y / 2, area_size.y / 2)
	var y = sample_terrain_height(Vector2(x, z))
	
	return Vector3(x, y, z)

func sample_terrain_height(xz: Vector2) -> float:
	"""Sample terrain height at given XZ position"""
	if not terrain or not terrain.mesh:
		return 0.0
	
	# Simplified: use raycast or height sampling
	# For now, return 0 (flat terrain assumption)
	return 0.0

func is_position_valid(pos: Vector3) -> bool:
	"""Check if position is valid for vegetation placement"""
	# Check distance from other objects
	for obj in placed_objects:
		if pos.distance_to(obj.global_position) < min_distance_between_objects:
			return false
	
	# Check slope if needed
	if avoid_steep_slopes:
		var normal = get_terrain_normal(pos)
		var angle = acos(normal.dot(Vector3.UP))
		if rad_to_deg(angle) > max_slope_angle:
			return false
	
	return true

func get_terrain_normal(pos: Vector3) -> Vector3:
	"""Get terrain normal at position"""
	# Simplified: return UP
	# In real implementation, sample from terrain mesh
	return Vector3.UP

func place_tree(pos: Vector3, rng: RandomNumberGenerator):
	"""Place a tree at the given position"""
	if tree_meshes.is_empty():
		return
	
	var tree = MeshInstance3D.new()
	tree.mesh = tree_meshes[rng.randi() % tree_meshes.size()]
	tree.position = pos
	
	# Random rotation
	tree.rotation.y = rng.randf() * TAU
	
	# Slight random scale
	var scale = rng.randf_range(0.8, 1.2)
	tree.scale = Vector3(scale, scale, scale)
	
	# Material for better visibility
	var material = StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.disable_receive_shadows = false
	material.shadow_to_opacity = false
	tree.material_override = material
	
	if cast_shadows:
		tree.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	
	add_child(tree)
	placed_objects.append(tree)

func place_rock(pos: Vector3, rng: RandomNumberGenerator):
	"""Place a rock at the given position"""
	if rock_meshes.is_empty():
		return
	
	var rock = MeshInstance3D.new()
	rock.mesh = rock_meshes[rng.randi() % rock_meshes.size()]
	rock.position = pos
	
	# Random rotation
	rock.rotation.y = rng.randf() * TAU
	rock.rotation.x = rng.randf_range(-0.2, 0.2)
	rock.rotation.z = rng.randf_range(-0.2, 0.2)
	
	# Random scale
	var scale = rng.randf_range(0.5, 1.5)
	rock.scale = Vector3(scale, scale * 0.8, scale)  # Flatter
	
	# Material
	var material = StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_PER_PIXEL
	material.roughness = 0.9
	material.metallic = 0.1
	rock.material_override = material
	
	if cast_shadows:
		rock.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	
	add_child(rock)
	placed_objects.append(rock)

func clear_vegetation():
	"""Remove all placed vegetation"""
	for obj in placed_objects:
		obj.queue_free()
	placed_objects.clear()

func _on_terrain_changed():
	"""Repopulate vegetation when terrain changes"""
	populate_vegetation()
