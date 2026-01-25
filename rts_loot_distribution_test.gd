extends Node2D
## Test script for RTS Loot Distribution System
## Run this in a 2D scene to visualize different distribution patterns

@export var test_pattern: RTSLootDistribution.DistributionPattern = RTSLootDistribution.DistributionPattern.CENTER_FOCUSED
@export var show_visualization: bool = true
@export var draw_map_bounds: bool = true

var loot_system: RTSLootDistribution
var camera: Camera2D


func _ready():
	# Setup camera
	camera = Camera2D.new()
	camera.zoom = Vector2(0.5, 0.5)
	add_child(camera)
	
	# Create loot distribution system
	loot_system = RTSLootDistribution.new()
	add_child(loot_system)
	
	# Configure for bigger map (ORC-135)
	loot_system.map_size = Vector2(2000, 2000)
	loot_system.distribution_pattern = test_pattern
	loot_system.total_loot_items = 150
	loot_system.center_weight = 2.5
	
	# Generate distribution
	loot_system.generate_loot_distribution()
	
	if show_visualization:
		loot_system.visualize_distribution()
	
	# Center camera on map
	camera.position = loot_system.map_size / 2
	
	# Force redraw
	queue_redraw()


func _draw():
	if not loot_system:
		return
	
	# Draw map bounds
	if draw_map_bounds:
		draw_rect(Rect2(Vector2.ZERO, loot_system.map_size), Color.DARK_GRAY, false, 5.0)
		
		# Draw center point
		var center = loot_system.map_size / 2
		draw_circle(center, 10, Color.YELLOW)
		
		# Draw center zone (25% radius)
		var center_radius = min(loot_system.map_size.x, loot_system.map_size.y) / 4
		draw_circle(center, center_radius, Color(1, 1, 0, 0.1))
		draw_arc(center, center_radius, 0, TAU, 64, Color(1, 1, 0, 0.5), 2.0)
	
	# Draw loot positions
	var loot_positions = loot_system.get_loot_positions()
	for loot in loot_positions:
		var color: Color
		var size: float
		
		match loot.tier:
			"common":
				color = Color.GREEN
				size = 5.0
			"rare":
				color = Color.BLUE
				size = 7.0
			"epic":
				color = Color.PURPLE
				size = 10.0
			_:
				color = Color.WHITE
				size = 5.0
		
		draw_circle(loot.position, size, color)
		# Draw outline
		draw_arc(loot.position, size, 0, TAU, 16, Color.WHITE, 1.0)


func _input(event):
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_1:
				_change_pattern(RTSLootDistribution.DistributionPattern.UNIFORM)
			KEY_2:
				_change_pattern(RTSLootDistribution.DistributionPattern.CENTER_FOCUSED)
			KEY_3:
				_change_pattern(RTSLootDistribution.DistributionPattern.EDGE_FOCUSED)
			KEY_4:
				_change_pattern(RTSLootDistribution.DistributionPattern.QUADRANT)
			KEY_5:
				_change_pattern(RTSLootDistribution.DistributionPattern.RING_PATTERN)
			KEY_R:
				_regenerate()
			KEY_PLUS, KEY_EQUAL:
				_adjust_center_weight(0.5)
			KEY_MINUS:
				_adjust_center_weight(-0.5)


func _change_pattern(pattern: RTSLootDistribution.DistributionPattern):
	loot_system.distribution_pattern = pattern
	loot_system.generate_loot_distribution()
	loot_system.visualize_distribution()
	queue_redraw()
	
	print("Switched to pattern: %s" % RTSLootDistribution.DistributionPattern.keys()[pattern])


func _regenerate():
	loot_system.seed_value = randi()
	loot_system.generate_loot_distribution()
	loot_system.visualize_distribution()
	queue_redraw()
	print("Regenerated with new seed")


func _adjust_center_weight(delta: float):
	loot_system.center_weight = max(1.0, loot_system.center_weight + delta)
	loot_system.generate_loot_distribution()
	loot_system.visualize_distribution()
	queue_redraw()
	print("Center weight adjusted to: %.1f" % loot_system.center_weight)


func _print_controls():
	print("\n=== RTS Loot Distribution Test Controls ===")
	print("1: Uniform Distribution")
	print("2: Center-Focused (Default)")
	print("3: Edge-Focused")
	print("4: Quadrant Distribution")
	print("5: Ring Pattern")
	print("R: Regenerate with new seed")
	print("+/-: Adjust center weight")
	print("==========================================\n")
