extends Node3D
class_name VegetationSystem

## VegetationSystem - Procedural vegetation placement for RTS maps
##
## Features:
## - Multiple vegetation types (grass, bushes, flowers, mushrooms)
## - Density-based placement with randomization
## - Performance-optimized with MultiMesh
## - Biome-based distribution

enum VegetationType {
	GRASS,
	BUSH,
	FLOWER,
	MUSHROOM,
	TALL_GRASS
}

## Vegetation density (plants per square meter)
@export var vegetation_density: float = 0.5

## Map size for vegetation placement
@export var map_size: Vector2 = Vector2(100, 100)

## Random seed for consistent generation
@export var random_seed: int = 12345

## Enable/disable different vegetation types
@export_group("Vegetation Types")
@export var enable_grass: bool = true
@export var enable_bushes: bool = true
@export var enable_flowers: bool = true
@export var enable_mushrooms: bool = true
@export var enable_tall_grass: bool = true

## Density multipliers for each type
@export_group("Density Settings")
@export var grass_density: float = 2.0
@export var bush_density: float = 0.3
@export var flower_density: float = 0.5
@export var mushroom_density: float = 0.2
@export var tall_grass_density: float = 1.0

## Reference to terrain for height sampling
@export var terrain: Node3D

var rng := RandomNumberGenerator.new()
var vegetation_instances: Dictionary = {}


func _ready() -> void:
	rng.seed = random_seed
	generate_vegetation()


## Generate all vegetation on the map
func generate_vegetation() -> void:
	print("VegetationSystem: Generating vegetation...")
	
	if enable_grass:
		_spawn_vegetation_type(VegetationType.GRASS, grass_density)
	
	if enable_bushes:
		_spawn_vegetation_type(VegetationType.BUSH, bush_density)
	
	if enable_flowers:
		_spawn_vegetation_type(VegetationType.FLOWER, flower_density)
	
	if enable_mushrooms:
		_spawn_vegetation_type(VegetationType.MUSHROOM, mushroom_density)
	
	if enable_tall_grass:
		_spawn_vegetation_type(VegetationType.TALL_GRASS, tall_grass_density)
	
	print("VegetationSystem: Vegetation generation complete!")


## Spawn a specific type of vegetation across the map
func _spawn_vegetation_type(type: VegetationType, density_multiplier: float) -> void:
	var type_name := VegetationType.keys()[type]
	var total_density := vegetation_density * density_multiplier
	var count := int(map_size.x * map_size.y * total_density)
	
	print("  Spawning %d %s instances..." % [count, type_name])
	
	# Create MultiMeshInstance3D for performance
	var multi_mesh_instance := MultiMeshInstance3D.new()
	multi_mesh_instance.name = type_name + "_MultiMesh"
	add_child(multi_mesh_instance)
	
	var multi_mesh := MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
	multi_mesh.instance_count = count
	multi_mesh.mesh = _create_vegetation_mesh(type)
	
	multi_mesh_instance.multimesh = multi_mesh
	
	# Place instances
	for i in range(count):
		var pos := _get_random_position()
		var transform := _create_vegetation_transform(pos, type)
		multi_mesh.set_instance_transform(i, transform)
	
	vegetation_instances[type] = multi_mesh_instance


## Get a random position on the map
func _get_random_position() -> Vector3:
	var x := rng.randf_range(-map_size.x / 2, map_size.x / 2)
	var z := rng.randf_range(-map_size.y / 2, map_size.y / 2)
	var y := _get_terrain_height(Vector2(x, z))
	
	return Vector3(x, y, z)


## Sample terrain height at a given position
func _get_terrain_height(pos: Vector2) -> float:
	if terrain and terrain.has_method("get_height_at"):
		return terrain.get_height_at(pos)
	return 0.0


## Create transform for vegetation instance
func _create_vegetation_transform(pos: Vector3, type: VegetationType) -> Transform3D:
	var transform := Transform3D()
	transform.origin = pos
	
	# Random rotation around Y axis
	var rotation_y := rng.randf_range(0, TAU)
	transform = transform.rotated(Vector3.UP, rotation_y)
	
	# Random scale variation
	var scale_variation := rng.randf_range(0.8, 1.2)
	var base_scale := _get_base_scale(type)
	transform = transform.scaled(Vector3.ONE * base_scale * scale_variation)
	
	# Slight random tilt for natural look
	if type in [VegetationType.GRASS, VegetationType.TALL_GRASS, VegetationType.FLOWER]:
		var tilt := rng.randf_range(-0.1, 0.1)
		transform = transform.rotated(Vector3.RIGHT, tilt)
	
	return transform


## Get base scale for vegetation type
func _get_base_scale(type: VegetationType) -> float:
	match type:
		VegetationType.GRASS:
			return 0.3
		VegetationType.BUSH:
			return 0.8
		VegetationType.FLOWER:
			return 0.4
		VegetationType.MUSHROOM:
			return 0.3
		VegetationType.TALL_GRASS:
			return 0.6
	return 1.0


## Create mesh for vegetation type (placeholder meshes)
func _create_vegetation_mesh(type: VegetationType) -> Mesh:
	var mesh: Mesh
	
	match type:
		VegetationType.GRASS:
			mesh = _create_grass_mesh()
		VegetationType.BUSH:
			mesh = _create_bush_mesh()
		VegetationType.FLOWER:
			mesh = _create_flower_mesh()
		VegetationType.MUSHROOM:
			mesh = _create_mushroom_mesh()
		VegetationType.TALL_GRASS:
			mesh = _create_tall_grass_mesh()
	
	return mesh


## Create simple grass mesh (crossed quads)
func _create_grass_mesh() -> Mesh:
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	
	var vertices := PackedVector3Array([
		# First blade
		Vector3(-0.1, 0, 0), Vector3(0.1, 0, 0),
		Vector3(0.1, 0.3, 0), Vector3(-0.1, 0.3, 0),
		# Second blade (crossed)
		Vector3(0, 0, -0.1), Vector3(0, 0, 0.1),
		Vector3(0, 0.3, 0.1), Vector3(0, 0.3, -0.1),
	])
	
	var indices := PackedInt32Array([
		0, 1, 2, 0, 2, 3,  # First blade
		4, 5, 6, 4, 6, 7,  # Second blade
	])
	
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_INDEX] = indices
	
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	
	# Add simple green material
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.2, 0.6, 0.2)
	material.cull_mode = StandardMaterial3D.CULL_DISABLED
	mesh.surface_set_material(0, material)
	
	return mesh


## Create bush mesh (sphere)
func _create_bush_mesh() -> Mesh:
	var mesh := SphereMesh.new()
	mesh.radius = 0.5
	mesh.height = 1.0
	mesh.radial_segments = 8
	mesh.rings = 4
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.1, 0.5, 0.1)
	mesh.material = material
	
	return mesh


## Create flower mesh (small sphere on stem)
func _create_flower_mesh() -> Mesh:
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	
	var vertices := PackedVector3Array([
		# Stem
		Vector3(-0.02, 0, 0), Vector3(0.02, 0, 0),
		Vector3(0.02, 0.3, 0), Vector3(-0.02, 0.3, 0),
	])
	
	var indices := PackedInt32Array([0, 1, 2, 0, 2, 3])
	
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_INDEX] = indices
	
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	
	# Colorful flower head will be added as separate mesh in production
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.9, 0.3, 0.5)
	mesh.surface_set_material(0, material)
	
	return mesh


## Create mushroom mesh (cylinder with cap)
func _create_mushroom_mesh() -> Mesh:
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.15
	mesh.bottom_radius = 0.05
	mesh.height = 0.2
	mesh.radial_segments = 8
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.8, 0.6, 0.4)
	mesh.material = material
	
	return mesh


## Create tall grass mesh (elongated crossed quads)
func _create_tall_grass_mesh() -> Mesh:
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	
	var vertices := PackedVector3Array([
		# First blade
		Vector3(-0.1, 0, 0), Vector3(0.1, 0, 0),
		Vector3(0.1, 0.6, 0), Vector3(-0.1, 0.6, 0),
		# Second blade
		Vector3(0, 0, -0.1), Vector3(0, 0, 0.1),
		Vector3(0, 0.6, 0.1), Vector3(0, 0.6, -0.1),
	])
	
	var indices := PackedInt32Array([
		0, 1, 2, 0, 2, 3,
		4, 5, 6, 4, 6, 7,
	])
	
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_INDEX] = indices
	
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.3, 0.7, 0.3)
	material.cull_mode = StandardMaterial3D.CULL_DISABLED
	mesh.surface_set_material(0, material)
	
	return mesh


## Clear all vegetation
func clear_vegetation() -> void:
	for child in get_children():
		child.queue_free()
	vegetation_instances.clear()


## Regenerate vegetation with new settings
func regenerate() -> void:
	clear_vegetation()
	generate_vegetation()
