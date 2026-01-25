extends Node3D

# Test scene for archer and crossbowman units
# Demonstrates unit training, combat, and projectiles

var archery_range: ArcheryRange
var test_target: Node3D
var spawned_units: Array = []

func _ready():
	setup_test_scene()
	
func setup_test_scene():
	# Create ground
	var ground = create_ground()
	add_child(ground)
	
	# Create archery range
	archery_range = create_archery_range()
	add_child(archery_range)
	
	# Create test target/dummy
	test_target = create_test_target()
	add_child(test_target)
	
	# Setup camera
	setup_camera()
	
	# Setup UI instructions
	print("=== Archery Units Test Scene ===")
	print("Press 'A' to train an Archer")
	print("Press 'C' to train a Crossbowman")
	print("Units will automatically attack the test target")
	print("================================")

func create_ground() -> StaticBody3D:
	var ground = StaticBody3D.new()
	
	var mesh_instance = MeshInstance3D.new()
	var plane_mesh = PlaneMesh.new()
	plane_mesh.size = Vector2(50, 50)
	mesh_instance.mesh = plane_mesh
	
	var material = StandardMaterial3D.new()
	material.albedo_color = Color(0.3, 0.5, 0.3)  # Green grass
	mesh_instance.set_surface_override_material(0, material)
	
	ground.add_child(mesh_instance)
	
	var collision = CollisionShape3D.new()
	var box_shape = BoxShape3D.new()
	box_shape.size = Vector3(50, 0.1, 50)
	collision.shape = box_shape
	ground.add_child(collision)
	
	return ground

func create_archery_range() -> ArcheryRange:
	var range_scene = preload("res://rts_game/buildings/archery_range.tscn")
	var range_instance = range_scene.instantiate()
	range_instance.position = Vector3(0, 0, 0)
	range_instance.set_faction(0)  # Human faction
	range_instance.unit_trained.connect(_on_unit_trained)
	return range_instance

func create_test_target() -> CharacterBody3D:
	# Create a simple target dummy
	var target = CharacterBody3D.new()
	target.position = Vector3(15, 0, 0)
	
	# Visual representation
	var mesh_instance = MeshInstance3D.new()
	var capsule_mesh = CapsuleMesh.new()
	capsule_mesh.height = 1.8
	capsule_mesh.radius = 0.4
	mesh_instance.mesh = capsule_mesh
	mesh_instance.position = Vector3(0, 0.9, 0)
	
	var material = StandardMaterial3D.new()
	material.albedo_color = Color(0.8, 0.2, 0.2)  # Red target
	mesh_instance.set_surface_override_material(0, material)
	
	target.add_child(mesh_instance)
	
	# Collision
	var collision = CollisionShape3D.new()
	var capsule_shape = CapsuleShape3D.new()
	capsule_shape.height = 1.8
	capsule_shape.radius = 0.4
	collision.position = Vector3(0, 0.9, 0)
	collision.shape = capsule_shape
	target.add_child(collision)
	
	# Add take_damage method
	target.set_script(preload("res://rts_game/scenes/test_target.gd"))
	
	return target

func setup_camera():
	var camera = Camera3D.new()
	camera.position = Vector3(10, 15, 20)
	camera.look_at(Vector3(5, 0, 0))
	add_child(camera)
	
	# Add directional light
	var light = DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-45, 45, 0)
	light.light_energy = 0.8
	add_child(light)

func _input(event):
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_A:
			print("Training Archer...")
			if archery_range:
				archery_range.train_archer()
		elif event.keycode == KEY_C:
			print("Training Crossbowman...")
			if archery_range:
				archery_range.train_crossbowman()
		elif event.keycode == KEY_T:
			# Manual spawn for testing
			spawn_test_archer()

func spawn_test_archer():
	var archer_scene = preload("res://rts_game/units/archer.tscn")
	var archer = archer_scene.instantiate()
	add_child(archer)
	archer.position = Vector3(3, 0, 3)
	archer.set_faction(0)
	archer.set_target(test_target)
	spawned_units.append(archer)
	print("Spawned test archer at position ", archer.position)

func _on_unit_trained(unit: Node3D):
	print("Unit trained: ", unit.name)
	spawned_units.append(unit)
	
	# Set the unit to attack the test target
	if test_target and unit.has_method("set_target"):
		unit.set_target(test_target)
		print("Unit ordered to attack target")

func _process(_delta):
	# Display stats
	if Engine.get_frames_drawn() % 60 == 0:  # Every 60 frames
		update_debug_info()

func update_debug_info():
	var alive_units = 0
	for unit in spawned_units:
		if is_instance_valid(unit):
			alive_units += 1
	
	if alive_units > 0:
		print("Active units: ", alive_units)
