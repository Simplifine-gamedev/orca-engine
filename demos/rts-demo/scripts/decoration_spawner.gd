extends Node3D
class_name DecorationSpawner

## DecorationSpawner - Places rocks, trees, and other large decorations
##
## Features:
## - Scatter rocks of various sizes
## - Place trees with variety
## - Avoid overlap with game objects
## - Support for custom decoration rules

enum DecorationType {
	ROCK_SMALL,
	ROCK_MEDIUM,
	ROCK_LARGE,
	TREE_PINE,
	TREE_OAK,
	TREE_BIRCH,
	BOULDER,
	DEAD_TREE
}

## Map size for decoration placement
@export var map_size: Vector2 = Vector2(100, 100)

## Random seed for consistent generation
@export var random_seed: int = 54321

## Enable/disable decoration types
@export_group("Decoration Types")
@export var enable_rocks: bool = true
@export var enable_trees: bool = true
@export var enable_boulders: bool = true
@export var enable_dead_trees: bool = true

## Density settings (instances per 100 square meters)
@export_group("Density Settings")
@export var rock_density: float = 5.0
@export var tree_density: float = 3.0
@export var boulder_density: float = 1.0
@export var dead_tree_density: float = 0.5

## Minimum distance between large decorations
@export var min_separation_distance: float = 3.0

## Reference to terrain
@export var terrain: Node3D

## Exclusion zones (areas to avoid)
@export var exclusion_zones: Array[Area3D] = []

var rng := RandomNumberGenerator.new()
var placed_positions: Array[Vector3] = []
var decoration_instances: Dictionary = {}


func _ready() -> void:
	rng.seed = random_seed
	generate_decorations()


## Generate all decorations
func generate_decorations() -> void:
	print("DecorationSpawner: Generating decorations...")
	
	if enable_rocks:
		_spawn_rocks()
	
	if enable_trees:
		_spawn_trees()
	
	if enable_boulders:
		_spawn_boulders()
	
	if enable_dead_trees:
		_spawn_dead_trees()
	
	print("DecorationSpawner: Placed %d decorations!" % placed_positions.size())


## Spawn rocks across the map
func _spawn_rocks() -> void:
	var area := map_size.x * map_size.y
	var count := int((area / 100.0) * rock_density)
	
	print("  Spawning %d rocks..." % count)
	
	for i in range(count):
		var rock_type := _get_random_rock_type()
		var pos := _find_valid_position(1.0)  # Rocks can be closer together
		
		if pos != Vector3.INF:
			_spawn_decoration(rock_type, pos)


## Spawn trees across the map
func _spawn_trees() -> void:
	var area := map_size.x * map_size.y
	var count := int((area / 100.0) * tree_density)
	
	print("  Spawning %d trees..." % count)
	
	for i in range(count):
		var tree_type := _get_random_tree_type()
		var pos := _find_valid_position(min_separation_distance)
		
		if pos != Vector3.INF:
			_spawn_decoration(tree_type, pos)


## Spawn boulders (large rocks)
func _spawn_boulders() -> void:
	var area := map_size.x * map_size.y
	var count := int((area / 100.0) * boulder_density)
	
	print("  Spawning %d boulders..." % count)
	
	for i in range(count):
		var pos := _find_valid_position(min_separation_distance * 1.5)
		
		if pos != Vector3.INF:
			_spawn_decoration(DecorationType.BOULDER, pos)


## Spawn dead trees
func _spawn_dead_trees() -> void:
	var area := map_size.x * map_size.y
	var count := int((area / 100.0) * dead_tree_density)
	
	print("  Spawning %d dead trees..." % count)
	
	for i in range(count):
		var pos := _find_valid_position(min_separation_distance)
		
		if pos != Vector3.INF:
			_spawn_decoration(DecorationType.DEAD_TREE, pos)


## Get random rock type
func _get_random_rock_type() -> DecorationType:
	var types := [
		DecorationType.ROCK_SMALL,
		DecorationType.ROCK_SMALL,  # More common
		DecorationType.ROCK_MEDIUM,
		DecorationType.ROCK_LARGE
	]
	return types[rng.randi() % types.size()]


## Get random tree type
func _get_random_tree_type() -> DecorationType:
	var types := [
		DecorationType.TREE_PINE,
		DecorationType.TREE_OAK,
		DecorationType.TREE_BIRCH
	]
	return types[rng.randi() % types.size()]


## Find a valid position that doesn't overlap with existing decorations
func _find_valid_position(min_distance: float) -> Vector3:
	const MAX_ATTEMPTS := 30
	
	for attempt in range(MAX_ATTEMPTS):
		var pos := _get_random_position()
		
		# Check if position is in exclusion zone
		if _is_in_exclusion_zone(pos):
			continue
		
		# Check distance to other decorations
		var valid := true
		for existing_pos in placed_positions:
			if pos.distance_to(existing_pos) < min_distance:
				valid = false
				break
		
		if valid:
			return pos
	
	return Vector3.INF  # Failed to find valid position


## Get random position on map
func _get_random_position() -> Vector3:
	var x := rng.randf_range(-map_size.x / 2, map_size.x / 2)
	var z := rng.randf_range(-map_size.y / 2, map_size.y / 2)
	var y := _get_terrain_height(Vector2(x, z))
	
	return Vector3(x, y, z)


## Sample terrain height
func _get_terrain_height(pos: Vector2) -> float:
	if terrain and terrain.has_method("get_height_at"):
		return terrain.get_height_at(pos)
	return 0.0


## Check if position is in exclusion zone
func _is_in_exclusion_zone(pos: Vector3) -> bool:
	for zone in exclusion_zones:
		if zone and zone.has_method("is_point_inside"):
			if zone.is_point_inside(pos):
				return true
	return false


## Spawn a decoration at the given position
func _spawn_decoration(type: DecorationType, pos: Vector3) -> void:
	var mesh_instance := MeshInstance3D.new()
	mesh_instance.name = DecorationType.keys()[type] + "_" + str(placed_positions.size())
	mesh_instance.mesh = _create_decoration_mesh(type)
	mesh_instance.position = pos
	
	# Random rotation
	mesh_instance.rotation.y = rng.randf_range(0, TAU)
	
	# Random scale variation
	var scale_range := _get_scale_range(type)
	var scale := rng.randf_range(scale_range.x, scale_range.y)
	mesh_instance.scale = Vector3.ONE * scale
	
	# Add slight random tilt for rocks
	if type in [DecorationType.ROCK_SMALL, DecorationType.ROCK_MEDIUM, 
				DecorationType.ROCK_LARGE, DecorationType.BOULDER]:
		mesh_instance.rotation.x = rng.randf_range(-0.2, 0.2)
		mesh_instance.rotation.z = rng.randf_range(-0.2, 0.2)
	
	add_child(mesh_instance)
	placed_positions.append(pos)
	
	if not decoration_instances.has(type):
		decoration_instances[type] = []
	decoration_instances[type].append(mesh_instance)


## Get scale range for decoration type
func _get_scale_range(type: DecorationType) -> Vector2:
	match type:
		DecorationType.ROCK_SMALL:
			return Vector2(0.5, 0.8)
		DecorationType.ROCK_MEDIUM:
			return Vector2(1.0, 1.5)
		DecorationType.ROCK_LARGE:
			return Vector2(1.8, 2.5)
		DecorationType.TREE_PINE:
			return Vector2(2.5, 3.5)
		DecorationType.TREE_OAK:
			return Vector2(2.0, 3.0)
		DecorationType.TREE_BIRCH:
			return Vector2(2.5, 3.2)
		DecorationType.BOULDER:
			return Vector2(3.0, 4.5)
		DecorationType.DEAD_TREE:
			return Vector2(2.0, 2.8)
	return Vector2(1.0, 1.0)


## Create mesh for decoration type
func _create_decoration_mesh(type: DecorationType) -> Mesh:
	match type:
		DecorationType.ROCK_SMALL, DecorationType.ROCK_MEDIUM, DecorationType.ROCK_LARGE:
			return _create_rock_mesh()
		DecorationType.TREE_PINE:
			return _create_pine_tree_mesh()
		DecorationType.TREE_OAK:
			return _create_oak_tree_mesh()
		DecorationType.TREE_BIRCH:
			return _create_birch_tree_mesh()
		DecorationType.BOULDER:
			return _create_boulder_mesh()
		DecorationType.DEAD_TREE:
			return _create_dead_tree_mesh()
	
	return SphereMesh.new()  # Fallback


## Create rock mesh (irregular sphere)
func _create_rock_mesh() -> Mesh:
	var mesh := SphereMesh.new()
	mesh.radius = 0.5
	mesh.height = 1.0
	mesh.radial_segments = 6
	mesh.rings = 4
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.5, 0.5, 0.5)
	material.roughness = 0.9
	mesh.material = material
	
	return mesh


## Create pine tree mesh (cone + cylinder trunk)
func _create_pine_tree_mesh() -> Mesh:
	# This is a simplified representation
	# In production, use proper 3D models
	var mesh := ArrayMesh.new()
	
	# Trunk
	var trunk := CylinderMesh.new()
	trunk.top_radius = 0.2
	trunk.bottom_radius = 0.2
	trunk.height = 2.0
	
	var trunk_material := StandardMaterial3D.new()
	trunk_material.albedo_color = Color(0.4, 0.3, 0.2)
	trunk.material = trunk_material
	
	# Foliage (cone)
	var foliage := ConeMesh.new()
	foliage.radius_top = 0.0
	foliage.radius_bottom = 1.5
	foliage.height = 3.0
	
	var foliage_material := StandardMaterial3D.new()
	foliage_material.albedo_color = Color(0.1, 0.4, 0.1)
	foliage.material = foliage_material
	
	return foliage  # Return foliage as main mesh


## Create oak tree mesh (sphere + trunk)
func _create_oak_tree_mesh() -> Mesh:
	var mesh := SphereMesh.new()
	mesh.radius = 1.5
	mesh.height = 3.0
	mesh.radial_segments = 8
	mesh.rings = 6
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.2, 0.5, 0.2)
	mesh.material = material
	
	return mesh


## Create birch tree mesh (thin cylinder + spherical foliage)
func _create_birch_tree_mesh() -> Mesh:
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.15
	mesh.bottom_radius = 0.2
	mesh.height = 4.0
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.9, 0.9, 0.85)
	mesh.material = material
	
	return mesh


## Create boulder mesh (large irregular rock)
func _create_boulder_mesh() -> Mesh:
	var mesh := SphereMesh.new()
	mesh.radius = 1.0
	mesh.height = 1.5
	mesh.radial_segments = 8
	mesh.rings = 6
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.4, 0.4, 0.4)
	material.roughness = 1.0
	mesh.material = material
	
	return mesh


## Create dead tree mesh (thin trunk, no foliage)
func _create_dead_tree_mesh() -> Mesh:
	var mesh := CylinderMesh.new()
	mesh.top_radius = 0.1
	mesh.bottom_radius = 0.3
	mesh.height = 3.0
	
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.3, 0.25, 0.2)
	mesh.material = material
	
	return mesh


## Clear all decorations
func clear_decorations() -> void:
	for child in get_children():
		child.queue_free()
	placed_positions.clear()
	decoration_instances.clear()


## Regenerate decorations
func regenerate() -> void:
	clear_decorations()
	generate_decorations()
