extends Node
class_name WallSystem

## Wall Building System with improved UX
## Improvements:
## - Right-click to cancel (instead of ESC)
## - Visual feedback during placement
## - Cost preview before confirming
## - Highlight valid placement areas

signal wall_placed(position: Vector2)
signal wall_cancelled
signal placement_mode_changed(active: bool)

# Configuration
@export var wall_cost: int = 50
@export var wall_health: int = 500
@export var placement_color_valid: Color = Color(0, 1, 0, 0.5)  # Green with alpha
@export var placement_color_invalid: Color = Color(1, 0, 0, 0.5)  # Red with alpha
@export var grid_size: int = 32  # Snap to grid

# State
var is_placing: bool = false
var current_placement_start: Vector2
var current_placement_end: Vector2
var placement_valid: bool = false
var placement_segments: Array[Vector2] = []
var show_tutorial: bool = true  # Show tutorial on first use

# References
var preview_nodes: Array[Node2D] = []
var collision_detector: Area2D
var ui_panel: Control

func _ready():
	# Initialize collision detector for valid placement checking
	setup_collision_detector()
	
	# Load tutorial state from user settings
	show_tutorial = _should_show_tutorial()

func _input(event: InputEvent):
	if not is_placing:
		return
	
	# RIGHT-CLICK TO CANCEL (improved UX)
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT and event.pressed:
			cancel_placement()
			return
		
		# Left click to confirm placement
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			if placement_valid:
				confirm_placement()
			else:
				show_invalid_placement_feedback()
	
	# Mouse movement - update preview
	if event is InputEventMouseMotion:
		update_placement_preview(event.position)
	
	# ESC as backup cancel option
	if event.is_action_pressed("ui_cancel"):
		cancel_placement()

func start_placement():
	"""Begin wall placement mode with tutorial if first time"""
	if show_tutorial:
		show_first_time_tutorial()
	
	is_placing = true
	placement_mode_changed.emit(true)
	
	# Show cost preview in UI
	if ui_panel:
		ui_panel.show_cost_preview(wall_cost)
	
	print("Wall placement started - Right-click to cancel, Left-click to place")

func cancel_placement():
	"""Cancel wall placement (RIGHT-CLICK or ESC)"""
	is_placing = false
	clear_preview()
	placement_mode_changed.emit(false)
	wall_cancelled.emit()
	
	if ui_panel:
		ui_panel.hide_cost_preview()
	
	print("Wall placement cancelled")

func update_placement_preview(mouse_pos: Vector2):
	"""Update visual feedback during placement"""
	if not is_placing:
		return
	
	# Snap to grid
	var snapped_pos = snap_to_grid(mouse_pos)
	current_placement_end = snapped_pos
	
	# Check if placement is valid
	placement_valid = check_valid_placement(snapped_pos)
	
	# Update visual preview
	update_preview_visuals(snapped_pos, placement_valid)
	
	# Highlight valid placement areas
	update_placement_area_highlights(snapped_pos)

func confirm_placement():
	"""Place the wall after validation"""
	if not placement_valid:
		return
	
	# Create wall instance
	var wall = create_wall_instance(current_placement_end)
	
	# Emit signal for game logic
	wall_placed.emit(current_placement_end)
	
	# Reset placement mode
	is_placing = false
	clear_preview()
	placement_mode_changed.emit(false)
	
	if ui_panel:
		ui_panel.hide_cost_preview()
	
	print("Wall placed at: ", current_placement_end)

func snap_to_grid(pos: Vector2) -> Vector2:
	"""Snap position to grid for clean placement"""
	return Vector2(
		floor(pos.x / grid_size) * grid_size,
		floor(pos.y / grid_size) * grid_size
	)

func check_valid_placement(pos: Vector2) -> bool:
	"""Check if placement position is valid"""
	# Check if position is within map bounds
	if not is_within_map_bounds(pos):
		return false
	
	# Check for collisions with existing structures
	if has_collision_at(pos):
		return false
	
	# Check for terrain restrictions (e.g., water, cliffs)
	if not is_valid_terrain(pos):
		return false
	
	return true

func is_within_map_bounds(pos: Vector2) -> bool:
	"""Check if position is within playable map area"""
	# TODO: Get actual map bounds from game manager
	var map_size = Vector2(2000, 2000)
	return pos.x >= 0 and pos.x < map_size.x and pos.y >= 0 and pos.y < map_size.y

func has_collision_at(pos: Vector2) -> bool:
	"""Check if there's a collision at the given position"""
	if not collision_detector:
		return false
	
	collision_detector.global_position = pos
	return collision_detector.has_overlapping_bodies() or collision_detector.has_overlapping_areas()

func is_valid_terrain(pos: Vector2) -> bool:
	"""Check if terrain type allows wall placement"""
	# TODO: Implement terrain checking with actual game terrain system
	# For now, assume all terrain is valid
	return true

func update_preview_visuals(pos: Vector2, valid: bool):
	"""Update the visual preview of wall placement"""
	# Clear existing preview
	clear_preview()
	
	# Create preview sprite/mesh
	var preview = Sprite2D.new()
	preview.name = "WallPreview"
	preview.position = pos
	preview.modulate = placement_color_valid if valid else placement_color_invalid
	
	# TODO: Load actual wall texture/sprite
	# preview.texture = preload("res://assets/walls/wall_preview.png")
	
	add_child(preview)
	preview_nodes.append(preview)

func update_placement_area_highlights(center_pos: Vector2):
	"""Highlight valid placement areas around cursor"""
	# Show valid placement grid around cursor
	var highlight_radius = 3  # Grid cells to highlight
	
	for x in range(-highlight_radius, highlight_radius + 1):
		for y in range(-highlight_radius, highlight_radius + 1):
			var grid_pos = center_pos + Vector2(x * grid_size, y * grid_size)
			if check_valid_placement(grid_pos):
				create_highlight_indicator(grid_pos)

func create_highlight_indicator(pos: Vector2):
	"""Create a visual indicator for valid placement area"""
	var indicator = ColorRect.new()
	indicator.size = Vector2(grid_size, grid_size)
	indicator.position = pos
	indicator.color = Color(0, 1, 0, 0.2)  # Light green
	indicator.name = "PlacementHighlight"
	
	add_child(indicator)
	preview_nodes.append(indicator)

func clear_preview():
	"""Clear all preview visuals"""
	for node in preview_nodes:
		node.queue_free()
	preview_nodes.clear()

func create_wall_instance(pos: Vector2) -> Node2D:
	"""Create the actual wall instance"""
	# TODO: Load actual wall scene
	# var wall_scene = preload("res://game/buildings/Wall.tscn")
	# var wall = wall_scene.instantiate()
	
	var wall = StaticBody2D.new()
	wall.name = "Wall"
	wall.position = pos
	
	# Add collision shape
	var collision = CollisionShape2D.new()
	var shape = RectangleShape2D.new()
	shape.size = Vector2(grid_size, grid_size)
	collision.shape = shape
	wall.add_child(collision)
	
	# Add visual sprite
	var sprite = Sprite2D.new()
	sprite.name = "Sprite"
	# TODO: Load actual wall texture
	# sprite.texture = preload("res://assets/walls/wall.png")
	wall.add_child(sprite)
	
	# Add to scene
	get_parent().add_child(wall)
	
	return wall

func setup_collision_detector():
	"""Setup area detector for collision checking"""
	collision_detector = Area2D.new()
	collision_detector.name = "CollisionDetector"
	
	var collision_shape = CollisionShape2D.new()
	var shape = RectangleShape2D.new()
	shape.size = Vector2(grid_size, grid_size)
	collision_shape.shape = shape
	
	collision_detector.add_child(collision_shape)
	add_child(collision_detector)
	
	# Set collision layers for building detection
	collision_detector.collision_layer = 0
	collision_detector.collision_mask = 1  # Detect buildings on layer 1

func show_first_time_tutorial():
	"""Show tutorial tooltip on first wall build"""
	if ui_panel:
		ui_panel.show_tutorial(
			"Wall Building",
			"Left-click to place walls\nRight-click to cancel\nWalls cost %d resources" % wall_cost
		)
	
	# Mark tutorial as shown
	show_tutorial = false
	_save_tutorial_state()

func show_invalid_placement_feedback():
	"""Show feedback when trying to place in invalid location"""
	if ui_panel:
		ui_panel.show_error("Cannot place wall here!")
	
	# TODO: Play error sound
	print("Invalid wall placement!")

func _should_show_tutorial() -> bool:
	"""Check if tutorial should be shown"""
	# TODO: Load from user settings/save file
	# For now, always show on first use
	return true

func _save_tutorial_state():
	"""Save that tutorial has been shown"""
	# TODO: Save to user settings/save file
	pass

func set_ui_panel(panel: Control):
	"""Set reference to UI panel for updates"""
	ui_panel = panel

func get_placement_cost() -> int:
	"""Get cost of placing a wall"""
	return wall_cost

func can_afford_wall(resources: int) -> bool:
	"""Check if player can afford to place a wall"""
	return resources >= wall_cost
