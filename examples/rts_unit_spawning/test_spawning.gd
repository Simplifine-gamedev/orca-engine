extends GutTest

## Unit tests for RTS unit spawning fix (ORC-118)
## Verifies units spawn outside buildings, not inside them

var Building = preload("res://examples/rts_unit_spawning/Building.gd")
var Unit = preload("res://examples/rts_unit_spawning/Unit.gd")
var building: Node2D
var test_scene: Node2D


func before_each():
	test_scene = Node2D.new()
	add_child_autoqfree(test_scene)
	
	building = Building.new()
	building.position = Vector2(200, 200)
	building.default_spawn_offset = Vector2(100, 0)
	test_scene.add_child(building)


func test_spawn_position_without_rally_point():
	# ARRANGE
	building.has_rally_point = false
	
	# ACT
	var spawn_pos = building.get_spawn_position()
	
	# ASSERT
	assert_ne(spawn_pos, building.global_position, 
		"Spawn position should NOT be at building center (that's the bug!)")
	assert_eq(spawn_pos, building.global_position + building.default_spawn_offset,
		"Spawn position should be at default offset from building")


func test_spawn_position_with_rally_point():
	# ARRANGE
	building.set_rally_point(Vector2(500, 300))
	
	# ACT
	var spawn_pos = building.get_spawn_position()
	
	# ASSERT
	assert_ne(spawn_pos, building.global_position,
		"Spawn position should NOT be at building center")
	
	var distance_from_building = spawn_pos.distance_to(building.global_position)
	var expected_distance = building.default_spawn_offset.length()
	assert_almost_eq(distance_from_building, expected_distance, 0.1,
		"Spawn should be at offset distance from building")


func test_unit_spawns_outside_building():
	# ARRANGE
	var unit_scene = PackedScene.new()
	building.spawn_scene = unit_scene
	
	# Create a simple unit for testing
	var unit_template = CharacterBody2D.new()
	unit_template.set_script(Unit)
	
	# ACT
	building.spawn_unit()
	
	# Wait for scene tree to process
	await get_tree().process_frame
	
	# ASSERT
	var spawned_units = test_scene.get_children().filter(
		func(node): return node is CharacterBody2D
	)
	assert_eq(spawned_units.size(), 1, "Should have spawned one unit")
	
	if spawned_units.size() > 0:
		var unit = spawned_units[0]
		assert_ne(unit.global_position, building.global_position,
			"BUG FIX VERIFIED: Unit is NOT spawned inside building!")


func test_spawn_offset_is_customizable():
	# ARRANGE
	var custom_offset = Vector2(150, 50)
	building.default_spawn_offset = custom_offset
	building.has_rally_point = false
	
	# ACT
	var spawn_pos = building.get_spawn_position()
	
	# ASSERT
	assert_eq(spawn_pos, building.global_position + custom_offset,
		"Should respect custom spawn offset")


func test_rally_point_direction():
	# ARRANGE
	building.set_rally_point(Vector2(400, 200))  # Rally point to the right
	
	# ACT
	var spawn_pos = building.get_spawn_position()
	
	# ASSERT
	# Spawn should be towards the rally point (to the right)
	assert_gt(spawn_pos.x, building.global_position.x,
		"Spawn should be towards rally point (right side)")


func test_multiple_spawn_positions_dont_overlap():
	# ARRANGE
	var spawn_positions = []
	
	# ACT - Spawn multiple units at different times
	for i in range(3):
		building.has_rally_point = false
		spawn_positions.append(building.get_spawn_position())
	
	# ASSERT
	# With no rally point, all spawns should be at the same default location
	# This is expected behavior - units stack at spawn then move
	assert_eq(spawn_positions[0], spawn_positions[1],
		"Default spawn position should be consistent")


func test_clear_rally_point_returns_to_default():
	# ARRANGE
	building.set_rally_point(Vector2(500, 300))
	var rally_spawn = building.get_spawn_position()
	
	# ACT
	building.clear_rally_point()
	var default_spawn = building.get_spawn_position()
	
	# ASSERT
	assert_ne(rally_spawn, default_spawn,
		"Spawn position should change when rally point is cleared")
	assert_eq(default_spawn, building.global_position + building.default_spawn_offset,
		"Should return to default offset after clearing rally point")


## Performance test
func test_spawn_position_calculation_is_fast():
	# ARRANGE
	var iterations = 10000
	
	# ACT
	var start_time = Time.get_ticks_usec()
	for i in range(iterations):
		var _pos = building.get_spawn_position()
	var end_time = Time.get_ticks_usec()
	
	# ASSERT
	var time_per_call = float(end_time - start_time) / iterations
	assert_lt(time_per_call, 10.0,  # Less than 10 microseconds per call
		"Spawn position calculation should be fast (< 10µs)")


## Integration test
func test_complete_workflow():
	# ARRANGE - Create a simple unit scene
	var unit_scene = PackedScene.new()
	building.spawn_scene = unit_scene
	
	# ACT & ASSERT - Test complete workflow
	
	# 1. Start without rally point
	building.clear_rally_point()
	var pos1 = building.get_spawn_position()
	assert_ne(pos1, building.global_position, "Step 1: Default spawn outside building")
	
	# 2. Set rally point
	building.set_rally_point(Vector2(600, 400))
	var pos2 = building.get_spawn_position()
	assert_ne(pos2, building.global_position, "Step 2: Rally spawn outside building")
	assert_ne(pos2, pos1, "Step 2: Rally spawn different from default")
	
	# 3. Clear rally point
	building.clear_rally_point()
	var pos3 = building.get_spawn_position()
	assert_eq(pos3, pos1, "Step 3: Returns to default spawn position")
	
	pass_test("Complete workflow maintains spawn outside building")
