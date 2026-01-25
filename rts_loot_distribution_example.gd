extends Node
## RTS Map Loot Distribution System
## 
## This script provides configurable loot distribution patterns for RTS maps.
## Addresses ORC-135: Allows control over whether loot concentrates in the center
## or distributes more evenly across the map.

class_name RTSLootDistribution

## Distribution pattern for loot placement
enum DistributionPattern {
	UNIFORM,           ## Even distribution across the map
	CENTER_FOCUSED,    ## Higher concentration in the center (default based on feedback)
	EDGE_FOCUSED,      ## Higher concentration at edges
	QUADRANT,          ## Distributed in quadrants
	RING_PATTERN       ## Concentric rings with varying density
}

## Configuration for loot distribution
@export_group("Loot Distribution Settings")
@export var distribution_pattern: DistributionPattern = DistributionPattern.CENTER_FOCUSED
@export var map_size: Vector2 = Vector2(1000, 1000)  ## Increased map size per ORC-135
@export var total_loot_items: int = 100
@export var center_weight: float = 2.5  ## How much to favor center (1.0 = no bias)
@export var min_distance_between_loot: float = 50.0
@export var seed_value: int = 0  ## 0 for random seed

@export_group("Loot Tiers")
@export var common_loot_percentage: float = 60.0
@export var rare_loot_percentage: float = 30.0
@export var epic_loot_percentage: float = 10.0

## Generated loot positions
var loot_positions: Array[Dictionary] = []


func _ready():
	if seed_value != 0:
		randomize()
		seed(seed_value)
	else:
		randomize()
	
	generate_loot_distribution()


## Generate loot positions based on the selected distribution pattern
func generate_loot_distribution() -> void:
	loot_positions.clear()
	
	match distribution_pattern:
		DistributionPattern.UNIFORM:
			_generate_uniform_distribution()
		DistributionPattern.CENTER_FOCUSED:
			_generate_center_focused_distribution()
		DistributionPattern.EDGE_FOCUSED:
			_generate_edge_focused_distribution()
		DistributionPattern.QUADRANT:
			_generate_quadrant_distribution()
		DistributionPattern.RING_PATTERN:
			_generate_ring_distribution()
	
	_assign_loot_tiers()
	
	print("Generated %d loot positions using %s pattern" % [
		loot_positions.size(),
		DistributionPattern.keys()[distribution_pattern]
	])


## Uniform distribution - completely even across the map
func _generate_uniform_distribution() -> void:
	var attempts = 0
	var max_attempts = total_loot_items * 10
	
	while loot_positions.size() < total_loot_items and attempts < max_attempts:
		var pos = Vector2(
			randf_range(0, map_size.x),
			randf_range(0, map_size.y)
		)
		
		if _is_valid_position(pos):
			loot_positions.append({
				"position": pos,
				"tier": "common"  # Will be reassigned later
			})
		
		attempts += 1


## Center-focused distribution - incentivizes exploration toward center
## This is the default pattern based on Haridzieko's positive feedback
func _generate_center_focused_distribution() -> void:
	var center = map_size / 2
	var max_distance = center.length()
	var attempts = 0
	var max_attempts = total_loot_items * 10
	
	while loot_positions.size() < total_loot_items and attempts < max_attempts:
		# Generate position with bias toward center
		var angle = randf() * TAU
		# Use power distribution to favor center
		var distance_factor = pow(randf(), center_weight)
		var distance = distance_factor * max_distance
		
		var pos = center + Vector2(cos(angle), sin(angle)) * distance
		
		# Clamp to map bounds
		pos.x = clamp(pos.x, 0, map_size.x)
		pos.y = clamp(pos.y, 0, map_size.y)
		
		if _is_valid_position(pos):
			loot_positions.append({
				"position": pos,
				"tier": "common"
			})
		
		attempts += 1


## Edge-focused distribution - rewards exploration of map edges
func _generate_edge_focused_distribution() -> void:
	var center = map_size / 2
	var max_distance = center.length()
	var attempts = 0
	var max_attempts = total_loot_items * 10
	
	while loot_positions.size() < total_loot_items and attempts < max_attempts:
		# Generate position with bias toward edges
		var angle = randf() * TAU
		# Invert the distribution to favor edges
		var distance_factor = 1.0 - pow(randf(), center_weight)
		var distance = distance_factor * max_distance
		
		var pos = center + Vector2(cos(angle), sin(angle)) * distance
		
		pos.x = clamp(pos.x, 0, map_size.x)
		pos.y = clamp(pos.y, 0, map_size.y)
		
		if _is_valid_position(pos):
			loot_positions.append({
				"position": pos,
				"tier": "common"
			})
		
		attempts += 1


## Quadrant-based distribution - strategic zones
func _generate_quadrant_distribution() -> void:
	var items_per_quadrant = total_loot_items / 4
	var quadrants = [
		Rect2(0, 0, map_size.x / 2, map_size.y / 2),
		Rect2(map_size.x / 2, 0, map_size.x / 2, map_size.y / 2),
		Rect2(0, map_size.y / 2, map_size.x / 2, map_size.y / 2),
		Rect2(map_size.x / 2, map_size.y / 2, map_size.x / 2, map_size.y / 2)
	]
	
	for quadrant in quadrants:
		var placed_in_quadrant = 0
		var attempts = 0
		var max_attempts = items_per_quadrant * 10
		
		while placed_in_quadrant < items_per_quadrant and attempts < max_attempts:
			var pos = Vector2(
				randf_range(quadrant.position.x, quadrant.position.x + quadrant.size.x),
				randf_range(quadrant.position.y, quadrant.position.y + quadrant.size.y)
			)
			
			if _is_valid_position(pos):
				loot_positions.append({
					"position": pos,
					"tier": "common"
				})
				placed_in_quadrant += 1
			
			attempts += 1


## Ring pattern distribution - concentric zones with varying density
func _generate_ring_distribution() -> void:
	var center = map_size / 2
	var max_radius = center.length()
	var num_rings = 5
	var items_per_ring = total_loot_items / num_rings
	
	for ring_index in range(num_rings):
		var inner_radius = (ring_index / float(num_rings)) * max_radius
		var outer_radius = ((ring_index + 1) / float(num_rings)) * max_radius
		
		var placed_in_ring = 0
		var attempts = 0
		var max_attempts = items_per_ring * 10
		
		while placed_in_ring < items_per_ring and attempts < max_attempts:
			var angle = randf() * TAU
			var radius = randf_range(inner_radius, outer_radius)
			var pos = center + Vector2(cos(angle), sin(angle)) * radius
			
			pos.x = clamp(pos.x, 0, map_size.x)
			pos.y = clamp(pos.y, 0, map_size.y)
			
			if _is_valid_position(pos):
				loot_positions.append({
					"position": pos,
					"tier": "common"
				})
				placed_in_ring += 1
			
			attempts += 1


## Check if a position is valid (not too close to existing loot)
func _is_valid_position(pos: Vector2) -> bool:
	for loot in loot_positions:
		if pos.distance_to(loot.position) < min_distance_between_loot:
			return false
	return true


## Assign loot tiers based on configured percentages
func _assign_loot_tiers() -> void:
	# Shuffle to randomize tier assignment
	loot_positions.shuffle()
	
	var num_common = int(total_loot_items * common_loot_percentage / 100.0)
	var num_rare = int(total_loot_items * rare_loot_percentage / 100.0)
	var num_epic = total_loot_items - num_common - num_rare
	
	var index = 0
	
	# Assign common
	for i in range(num_common):
		if index < loot_positions.size():
			loot_positions[index].tier = "common"
			index += 1
	
	# Assign rare
	for i in range(num_rare):
		if index < loot_positions.size():
			loot_positions[index].tier = "rare"
			index += 1
	
	# Assign epic
	for i in range(num_epic):
		if index < loot_positions.size():
			loot_positions[index].tier = "epic"
			index += 1


## Get all loot positions
func get_loot_positions() -> Array[Dictionary]:
	return loot_positions


## Get loot positions by tier
func get_loot_by_tier(tier: String) -> Array[Dictionary]:
	var filtered: Array[Dictionary] = []
	for loot in loot_positions:
		if loot.tier == tier:
			filtered.append(loot)
	return filtered


## Visualize the distribution (for debugging)
func visualize_distribution() -> void:
	print("\n=== Loot Distribution Visualization ===")
	print("Pattern: %s" % DistributionPattern.keys()[distribution_pattern])
	print("Map Size: %v" % map_size)
	print("Total Loot: %d" % loot_positions.size())
	
	# Calculate center concentration
	var center = map_size / 2
	var center_radius = min(map_size.x, map_size.y) / 4
	var center_count = 0
	
	for loot in loot_positions:
		if loot.position.distance_to(center) < center_radius:
			center_count += 1
	
	var center_percentage = (center_count / float(loot_positions.size())) * 100.0
	print("Loot in center 25%% of map: %.1f%%" % center_percentage)
	
	# Tier distribution
	var tier_counts = {"common": 0, "rare": 0, "epic": 0}
	for loot in loot_positions:
		tier_counts[loot.tier] += 1
	
	print("\nTier Distribution:")
	print("  Common: %d (%.1f%%)" % [tier_counts.common, (tier_counts.common / float(loot_positions.size())) * 100])
	print("  Rare: %d (%.1f%%)" % [tier_counts.rare, (tier_counts.rare / float(loot_positions.size())) * 100])
	print("  Epic: %d (%.1f%%)" % [tier_counts.epic, (tier_counts.epic / float(loot_positions.size())) * 100])
	print("=====================================\n")
