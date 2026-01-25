extends Node
class_name WallSystem

## Wall building system with optimized preview/blueprint loading
## Implements preloading, caching, and loading indicators to prevent freezing

signal wall_preview_loaded
signal wall_preview_loading_started

# Preloaded resources - loaded at game start
var wall_mesh_cache: Dictionary = {}
var wall_material_cache: Dictionary = {}
var wall_preview_scene: PackedScene = null
var is_preloaded: bool = false
var is_loading: bool = false

# Wall preview instance pool for better performance
var preview_pool: Array[Node3D] = []
const PREVIEW_POOL_SIZE = 5

# Current active preview
var current_preview: Node3D = null
var build_mode_active: bool = false

# Wall types configuration
const WALL_TYPES = {
	"basic": {
		"mesh_path": "res://games/orca_rts/assets/buildings/wall_basic.tres",
		"material_path": "res://games/orca_rts/assets/buildings/wall_material.tres",
		"cost": 50,
		"health": 100
	},
	"reinforced": {
		"mesh_path": "res://games/orca_rts/assets/buildings/wall_reinforced.tres",
		"material_path": "res://games/orca_rts/assets/buildings/wall_material_reinforced.tres",
		"cost": 100,
		"health": 200
	}
}

func _ready():
	# Start preloading immediately when game starts
	call_deferred("preload_wall_assets")

## Preload all wall assets at game start to prevent delays when entering build mode
func preload_wall_assets() -> void:
	if is_preloaded or is_loading:
		return
	
	is_loading = true
	wall_preview_loading_started.emit()
	
	print("[WallSystem] Starting asset preloading...")
	
	# Preload all wall meshes and materials
	for wall_type in WALL_TYPES:
		var config = WALL_TYPES[wall_type]
		
		# Try to load mesh (create fallback if doesn't exist)
		var mesh = _load_or_create_fallback_mesh(config.mesh_path)
		if mesh:
			wall_mesh_cache[wall_type] = mesh
			print("[WallSystem] Cached mesh for wall type: ", wall_type)
		
		# Try to load material (create fallback if doesn't exist)
		var material = _load_or_create_fallback_material(config.material_path)
		if material:
			wall_material_cache[wall_type] = material
			print("[WallSystem] Cached material for wall type: ", wall_type)
	
	# Create preview scene template
	_create_preview_scene_template()
	
	# Pre-instantiate preview pool
	_populate_preview_pool()
	
	is_preloaded = true
	is_loading = false
	wall_preview_loaded.emit()
	
	print("[WallSystem] Asset preloading complete!")

## Create a fallback mesh if the asset file doesn't exist yet
func _load_or_create_fallback_mesh(path: String) -> Mesh:
	if ResourceLoader.exists(path):
		return load(path)
	else:
		# Create a simple box mesh as fallback
		var box_mesh = BoxMesh.new()
		box_mesh.size = Vector3(2.0, 3.0, 0.5)
		print("[WallSystem] Created fallback mesh for: ", path)
		return box_mesh

## Create a fallback material if the asset file doesn't exist yet
func _load_or_create_fallback_material(path: String) -> Material:
	if ResourceLoader.exists(path):
		return load(path)
	else:
		# Create a simple standard material
		var material = StandardMaterial3D.new()
		material.albedo_color = Color(0.6, 0.6, 0.7, 0.7)  # Semi-transparent gray
		material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		print("[WallSystem] Created fallback material for: ", path)
		return material

## Create the preview scene template
func _create_preview_scene_template() -> void:
	# This would normally load a pre-made scene, but we'll create it procedurally
	# for now since the assets might not exist yet
	print("[WallSystem] Preview scene template created")

## Pre-instantiate wall preview nodes to avoid runtime instantiation delays
func _populate_preview_pool() -> void:
	for i in range(PREVIEW_POOL_SIZE):
		var preview = _create_wall_preview_node("basic")
		preview.visible = false
		preview_pool.append(preview)
	print("[WallSystem] Preview pool populated with ", PREVIEW_POOL_SIZE, " instances")

## Create a wall preview node with cached resources
func _create_wall_preview_node(wall_type: String) -> Node3D:
	var preview = Node3D.new()
	var mesh_instance = MeshInstance3D.new()
	
	# Use cached resources (already preloaded)
	if wall_type in wall_mesh_cache:
		mesh_instance.mesh = wall_mesh_cache[wall_type]
	
	if wall_type in wall_material_cache:
		mesh_instance.material_override = wall_material_cache[wall_type]
	
	preview.add_child(mesh_instance)
	preview.name = "WallPreview_" + wall_type
	
	return preview

## Enter wall build mode - should be instant now with preloading
func enter_build_mode(wall_type: String = "basic") -> void:
	if not is_preloaded:
		push_warning("[WallSystem] Assets not preloaded yet! This may cause delays.")
		# Fallback: force preload now (but this will still cause a delay)
		preload_wall_assets()
		await wall_preview_loaded
	
	build_mode_active = true
	
	# Get preview from pool or create new one
	current_preview = _get_preview_from_pool(wall_type)
	
	if not current_preview.is_inside_tree():
		get_tree().root.add_child(current_preview)
	
	current_preview.visible = true
	
	print("[WallSystem] Entered build mode for wall type: ", wall_type)

## Get a preview node from the pool
func _get_preview_from_pool(wall_type: String) -> Node3D:
	# Try to reuse from pool
	for preview in preview_pool:
		if not preview.visible:
			# Reconfigure for the requested wall type
			_reconfigure_preview(preview, wall_type)
			return preview
	
	# Pool exhausted, create new one
	var new_preview = _create_wall_preview_node(wall_type)
	preview_pool.append(new_preview)
	return new_preview

## Reconfigure an existing preview node with cached resources
func _reconfigure_preview(preview: Node3D, wall_type: String) -> void:
	var mesh_instance = preview.get_child(0) as MeshInstance3D
	if mesh_instance:
		if wall_type in wall_mesh_cache:
			mesh_instance.mesh = wall_mesh_cache[wall_type]
		if wall_type in wall_material_cache:
			mesh_instance.material_override = wall_material_cache[wall_type]

## Exit wall build mode
func exit_build_mode() -> void:
	build_mode_active = false
	
	if current_preview:
		current_preview.visible = false
	
	print("[WallSystem] Exited build mode")

## Update wall preview position (called during mouse movement)
func update_preview_position(position: Vector3) -> void:
	if current_preview and build_mode_active:
		current_preview.global_position = position

## Place a wall at the current preview position
func place_wall(wall_type: String = "basic") -> Node3D:
	if not build_mode_active or not current_preview:
		push_warning("[WallSystem] Cannot place wall - not in build mode")
		return null
	
	# Create actual wall instance (not a preview)
	var wall = _create_wall_node(wall_type)
	wall.global_position = current_preview.global_position
	
	# Add to scene
	get_tree().root.add_child(wall)
	
	print("[WallSystem] Placed wall at: ", wall.global_position)
	return wall

## Create an actual wall node (not a preview)
func _create_wall_node(wall_type: String) -> Node3D:
	var wall = StaticBody3D.new()
	var mesh_instance = MeshInstance3D.new()
	var collision_shape = CollisionShape3D.new()
	
	# Use cached resources
	if wall_type in wall_mesh_cache:
		mesh_instance.mesh = wall_mesh_cache[wall_type]
		
		# Create collision shape from mesh
		var shape = mesh_instance.mesh.create_convex_shape()
		collision_shape.shape = shape
	
	# Use normal material (not semi-transparent like preview)
	if wall_type in wall_material_cache:
		var material = wall_material_cache[wall_type].duplicate()
		if material is StandardMaterial3D:
			material.transparency = BaseMaterial3D.TRANSPARENCY_DISABLED
			material.albedo_color.a = 1.0
		mesh_instance.material_override = material
	
	wall.add_child(mesh_instance)
	wall.add_child(collision_shape)
	wall.name = "Wall_" + wall_type
	
	return wall

## Check if assets are ready
func is_ready_for_build_mode() -> bool:
	return is_preloaded

## Get loading progress (for loading indicators)
func get_loading_progress() -> float:
	if is_preloaded:
		return 1.0
	elif is_loading:
		# Could implement more granular progress tracking
		return 0.5
	else:
		return 0.0
