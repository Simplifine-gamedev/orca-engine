extends StaticBody3D
class_name RTSBuilding

## RTS Building that spawns units properly without T-posing
##
## This demonstrates the correct way to spawn units:
## 1. Instantiate the unit scene
## 2. Set position BEFORE adding to tree
## 3. Add to tree (triggers _ready() which sets animation)
## 4. Optionally call spawn initialization

@export var unit_scene: PackedScene
@export var spawn_offset: Vector3 = Vector3(0, 0, 3)
@export var spawn_interval: float = 5.0
@export var max_queue: int = 5
@export var auto_spawn: bool = false

var spawn_queue: int = 0
var spawn_timer: float = 0.0
var rally_point: Vector3 = Vector3.ZERO

@onready var spawn_point: Marker3D = $SpawnPoint if has_node("SpawnPoint") else null


func _ready() -> void:
	# Set default rally point if not set
	if rally_point == Vector3.ZERO:
		rally_point = global_position + spawn_offset * 2
	
	# Start auto-spawning if enabled
	if auto_spawn and spawn_queue == 0:
		queue_unit()


func _process(delta: float) -> void:
	# Handle spawn queue
	if spawn_queue > 0:
		spawn_timer -= delta
		
		if spawn_timer <= 0:
			spawn_unit()
			spawn_timer = spawn_interval


func queue_unit() -> bool:
	"""Add a unit to the spawn queue"""
	if spawn_queue >= max_queue:
		print("Spawn queue full!")
		return false
	
	spawn_queue += 1
	
	# Start timer if this is first in queue
	if spawn_queue == 1:
		spawn_timer = spawn_interval
	
	print("Unit queued. Queue size: ", spawn_queue)
	return true


func spawn_unit() -> void:
	"""
	Spawn a unit with proper animation initialization.
	
	CRITICAL ORDER to prevent T-posing:
	1. Instantiate scene
	2. Set position/rotation BEFORE adding to tree
	3. Add to scene tree (triggers _ready() which initializes animation)
	4. Call any post-spawn setup
	"""
	
	if not unit_scene:
		push_error("No unit scene assigned to building!")
		return
	
	if spawn_queue <= 0:
		return
	
	# Calculate spawn position
	var spawn_pos = _get_spawn_position()
	
	# STEP 1: Instantiate the unit (but don't add to tree yet)
	var unit: RTSUnit = unit_scene.instantiate()
	
	if not unit:
		push_error("Failed to instantiate unit!")
		return
	
	# STEP 2: Set transform BEFORE adding to scene tree
	# This ensures the unit appears in the right place when _ready() is called
	unit.global_position = spawn_pos
	unit.rotation.y = rotation.y  # Face same direction as building
	
	# STEP 3: Add to scene tree
	# This triggers unit._ready() which initializes animations immediately
	# The animation system will start in idle state, preventing T-pose
	get_tree().current_scene.add_child(unit)
	
	# STEP 4: Post-spawn setup (optional)
	# The unit is already showing idle animation at this point
	unit.spawn_at_position(spawn_pos)  # Extra safety call
	
	# Send unit to rally point
	if rally_point != Vector3.ZERO:
		# Small delay before moving to ensure animation is fully initialized
		await get_tree().create_timer(0.1).timeout
		unit.move_to(rally_point)
	
	# Update queue
	spawn_queue -= 1
	
	print("Unit spawned at ", spawn_pos, " - Animation state initialized to IDLE")
	
	# Emit signal for other systems
	unit_spawned.emit(unit)
	
	# Continue spawning if queue not empty
	if spawn_queue > 0:
		spawn_timer = spawn_interval
	
	# Auto-queue next unit if enabled
	if auto_spawn and spawn_queue == 0:
		queue_unit()


func _get_spawn_position() -> Vector3:
	"""Calculate where to spawn the unit"""
	if spawn_point:
		return spawn_point.global_position
	else:
		return global_position + spawn_offset


func set_rally_point(point: Vector3) -> void:
	"""Set where spawned units should move to"""
	rally_point = point


## SIGNALS
signal unit_spawned(unit: RTSUnit)


## PUBLIC METHODS for testing/demo

func spawn_unit_immediately() -> void:
	"""Bypass queue and spawn unit right now (for testing)"""
	var temp_queue = spawn_queue
	spawn_queue = 1
	spawn_unit()
	spawn_queue = temp_queue


func spawn_multiple_units(count: int) -> void:
	"""Queue multiple units at once"""
	for i in range(count):
		if not queue_unit():
			break


## ALTERNATIVE SPAWN METHOD (Less reliable, can cause T-posing)
## This is what NOT to do - shown here for educational purposes

func spawn_unit_wrong_way() -> void:
	"""
	WRONG WAY to spawn units - causes T-posing!
	
	Problems:
	1. Adds to tree before setting position
	2. Doesn't wait for _ready() to complete
	3. Tries to set animation state before system is initialized
	"""
	if not unit_scene:
		return
	
	# PROBLEM: Instantiate and add to tree immediately
	var unit: RTSUnit = unit_scene.instantiate()
	get_tree().current_scene.add_child(unit)
	
	# PROBLEM: Set position AFTER adding to tree
	# The unit might render one frame at origin (0,0,0) in T-pose
	unit.global_position = _get_spawn_position()
	
	# PROBLEM: Try to set animation before _ready() completes
	# Animation system might not be initialized yet
	# This can fail silently, leaving unit in T-pose
	# unit._set_animation_state(RTSUnit.UnitState.IDLE)  # Won't work reliably!
	
	print("Unit spawned THE WRONG WAY - may T-pose on first frame!")
