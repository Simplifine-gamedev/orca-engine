extends Node3D
class_name HeightmapTerrain

## HeightmapTerrain - Procedural terrain generation with heightmap support
##
## Features:
## - Heightmap-based terrain generation
## - Multiple biomes
## - Terrain features (hills, cliffs, valleys)
## - Integration with vegetation and decoration systems

## Terrain size in world units
@export var terrain_size: Vector2 = Vector2(100, 100)

## Height scale multiplier
@export var height_scale: float = 10.0

## Terrain resolution (vertices per side)
@export var resolution: int = 100

## Noise seed for terrain generation
@export var noise_seed: int = 0

## Noise settings
@export_group("Noise Settings")
@export var noise_frequency: float = 0.05
@export var noise_octaves: int = 4
@export var noise_lacunarity: float = 2.0
@export var noise_persistence: float = 0.5

## Enable terrain features
@export_group("Terrain Features")
@export var enable_hills: bool = true
@export var enable_valleys: bool = true
@export var enable_plateaus: bool = false

## Biome settings
@export_group("Biomes")
@export var biome_variation: float = 0.3

var noise: FastNoiseLite
var heightmap: Array = []
var mesh_instance: MeshInstance3D


func _ready() -> void:
	_initialize_noise()
	generate_terrain()


## Initialize noise generator
func _initialize_noise() -> void:
	noise = FastNoiseLite.new()
	noise.seed = noise_seed
	noise.noise_type = FastNoiseLite.TYPE_PERLIN
	noise.frequency = noise_frequency
	noise.fractal_octaves = noise_octaves
	noise.fractal_lacunarity = noise_lacunarity
	noise.fractal_gain = noise_persistence


## Generate the terrain mesh
func generate_terrain() -> void:
	print("HeightmapTerrain: Generating terrain...")
	
	# Generate heightmap
	_generate_heightmap()
	
	# Create mesh from heightmap
	var mesh := _create_terrain_mesh()
	
	# Create or update mesh instance
	if not mesh_instance:
		mesh_instance = MeshInstance3D.new()
		mesh_instance.name = "TerrainMesh"
		add_child(mesh_instance)
	
	mesh_instance.mesh = mesh
	
	print("HeightmapTerrain: Terrain generation complete!")


## Generate heightmap data
func _generate_heightmap() -> void:
	heightmap.clear()
	heightmap.resize(resolution)
	
	for x in range(resolution):
		heightmap[x] = []
		heightmap[x].resize(resolution)
		
		for z in range(resolution):
			var world_x := (x / float(resolution) - 0.5) * terrain_size.x
			var world_z := (z / float(resolution) - 0.5) * terrain_size.y
			
			var height := _calculate_height_at(Vector2(world_x, world_z))
			heightmap[x][z] = height


## Calculate height at a world position
func _calculate_height_at(pos: Vector2) -> float:
	var base_height := noise.get_noise_2d(pos.x, pos.y) * height_scale
	
	# Add terrain features
	if enable_hills:
		base_height += _get_hill_height(pos)
	
	if enable_valleys:
		base_height += _get_valley_height(pos)
	
	if enable_plateaus:
		base_height = _apply_plateau_effect(base_height)
	
	return base_height


## Get hill contribution
func _get_hill_height(pos: Vector2) -> float:
	var hill_noise := noise.get_noise_2d(pos.x * 0.02, pos.y * 0.02)
	return max(0, hill_noise) * height_scale * 0.5


## Get valley contribution
func _get_valley_height(pos: Vector2) -> float:
	var valley_noise := noise.get_noise_2d(pos.x * 0.03 + 1000, pos.y * 0.03 + 1000)
	return min(0, valley_noise) * height_scale * 0.3


## Apply plateau effect
func _apply_plateau_effect(height: float) -> float:
	# Quantize height into levels
	var level_count := 5
	var level_height := height_scale / level_count
	return floor(height / level_height) * level_height


## Create terrain mesh from heightmap
func _create_terrain_mesh() -> Mesh:
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var uvs := PackedVector2Array()
	var colors := PackedColorArray()
	var indices := PackedInt32Array()
	
	# Generate vertices
	for x in range(resolution):
		for z in range(resolution):
			var world_x := (x / float(resolution) - 0.5) * terrain_size.x
			var world_z := (z / float(resolution) - 0.5) * terrain_size.y
			var height := heightmap[x][z]
			
			vertices.append(Vector3(world_x, height, world_z))
			
			# Calculate normal (approximate)
			var normal := _calculate_normal(x, z)
			normals.append(normal)
			
			# UV coordinates
			uvs.append(Vector2(x / float(resolution), z / float(resolution)))
			
			# Vertex color based on height (biome)
			var color := _get_terrain_color(height)
			colors.append(color)
	
	# Generate indices
	for x in range(resolution - 1):
		for z in range(resolution - 1):
			var i := x * resolution + z
			var i_right := (x + 1) * resolution + z
			var i_down := x * resolution + (z + 1)
			var i_diag := (x + 1) * resolution + (z + 1)
			
			# First triangle
			indices.append(i)
			indices.append(i_right)
			indices.append(i_diag)
			
			# Second triangle
			indices.append(i)
			indices.append(i_diag)
			indices.append(i_down)
	
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = indices
	
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	
	# Add material
	var material := StandardMaterial3D.new()
	material.vertex_color_use_as_albedo = true
	material.roughness = 0.8
	mesh.surface_set_material(0, material)
	
	return mesh


## Calculate normal at heightmap position
func _calculate_normal(x: int, z: int) -> Vector3:
	var h_center := heightmap[x][z]
	
	var h_left := h_center
	var h_right := h_center
	var h_up := h_center
	var h_down := h_center
	
	if x > 0:
		h_left = heightmap[x - 1][z]
	if x < resolution - 1:
		h_right = heightmap[x + 1][z]
	if z > 0:
		h_up = heightmap[x][z - 1]
	if z < resolution - 1:
		h_down = heightmap[x][z + 1]
	
	var step := terrain_size.x / resolution
	
	var dx := (h_right - h_left) / (step * 2.0)
	var dz := (h_down - h_up) / (step * 2.0)
	
	return Vector3(-dx, 1.0, -dz).normalized()


## Get terrain color based on height (biome)
func _get_terrain_color(height: float) -> Color:
	var normalized_height := (height + height_scale) / (height_scale * 2.0)
	
	# Water
	if normalized_height < 0.3:
		return Color(0.2, 0.4, 0.6)
	
	# Beach/Sand
	elif normalized_height < 0.35:
		return Color(0.8, 0.7, 0.5)
	
	# Grass
	elif normalized_height < 0.6:
		return Color(0.3, 0.6, 0.3)
	
	# Forest (darker green)
	elif normalized_height < 0.7:
		return Color(0.2, 0.5, 0.2)
	
	# Mountain (brown/gray)
	elif normalized_height < 0.85:
		return Color(0.5, 0.4, 0.3)
	
	# Snow
	else:
		return Color(0.9, 0.9, 0.95)


## Get height at a specific world position
func get_height_at(pos: Vector2) -> float:
	# Convert world position to heightmap coordinates
	var x := int((pos.x / terrain_size.x + 0.5) * resolution)
	var z := int((pos.y / terrain_size.y + 0.5) * resolution)
	
	# Clamp to valid range
	x = clampi(x, 0, resolution - 1)
	z = clampi(z, 0, resolution - 1)
	
	if heightmap.size() > x and heightmap[x].size() > z:
		return heightmap[x][z]
	
	# Fallback: calculate on the fly
	return _calculate_height_at(pos)


## Get biome type at position
func get_biome_at(pos: Vector2) -> String:
	var height := get_height_at(pos)
	var normalized_height := (height + height_scale) / (height_scale * 2.0)
	
	if normalized_height < 0.3:
		return "water"
	elif normalized_height < 0.35:
		return "beach"
	elif normalized_height < 0.6:
		return "grassland"
	elif normalized_height < 0.7:
		return "forest"
	elif normalized_height < 0.85:
		return "mountain"
	else:
		return "snow"


## Check if position is suitable for decoration placement
func is_decoration_suitable(pos: Vector2) -> bool:
	var biome := get_biome_at(pos)
	return biome in ["grassland", "forest"]


## Clear and regenerate terrain
func regenerate() -> void:
	_initialize_noise()
	generate_terrain()
