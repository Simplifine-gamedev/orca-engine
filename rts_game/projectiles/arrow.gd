extends Area3D
class_name Arrow

# Arrow projectile fired by archers
# Travels in straight line and deals damage on impact

var direction: Vector3 = Vector3.FORWARD
var speed: float = 20.0
var damage: float = 12.0
var faction_id: int = 0
var max_distance: float = 30.0
var traveled_distance: float = 0.0

@onready var mesh: MeshInstance3D = $ArrowMesh
@onready var trail: GPUParticles3D = $Trail

func _ready():
	setup_visuals()
	body_entered.connect(_on_body_entered)
	area_entered.connect(_on_area_entered)

func setup_visuals():
	# Create arrow mesh if not exists
	if not mesh:
		mesh = MeshInstance3D.new()
		add_child(mesh)
		
		var arrow_mesh = CylinderMesh.new()
		arrow_mesh.height = 0.8
		arrow_mesh.top_radius = 0.02
		arrow_mesh.bottom_radius = 0.02
		mesh.mesh = arrow_mesh
		
		# Rotate to point forward
		mesh.rotation.z = deg_to_rad(90)
		
		# Brown wood color
		var material = StandardMaterial3D.new()
		material.albedo_color = Color(0.4, 0.3, 0.2)
		mesh.set_surface_override_material(0, material)
	
	# Add arrowhead (cone)
	var arrowhead = MeshInstance3D.new()
	var cone_mesh = CylinderMesh.new()
	cone_mesh.height = 0.2
	cone_mesh.top_radius = 0.0
	cone_mesh.bottom_radius = 0.05
	arrowhead.mesh = cone_mesh
	arrowhead.position = Vector3(0.5, 0, 0)
	arrowhead.rotation.z = deg_to_rad(90)
	var metal_material = StandardMaterial3D.new()
	metal_material.albedo_color = Color(0.7, 0.7, 0.7)
	metal_material.metallic = 0.8
	arrowhead.set_surface_override_material(0, metal_material)
	add_child(arrowhead)

func set_direction(dir: Vector3, proj_speed: float, proj_damage: float, proj_faction: int):
	direction = dir.normalized()
	speed = proj_speed
	damage = proj_damage
	faction_id = proj_faction
	
	# Rotate to face direction
	if direction.length() > 0:
		look_at(global_position + direction)

func _physics_process(delta):
	var movement = direction * speed * delta
	global_position += movement
	traveled_distance += movement.length()
	
	# Destroy if traveled too far
	if traveled_distance >= max_distance:
		queue_free()

func _on_body_entered(body):
	if body.has_method("take_damage"):
		# Check if not same faction
		if body.has_method("get_faction"):
			if body.get_faction() != faction_id:
				body.take_damage(damage)
				create_impact_effect()
				queue_free()
		else:
			body.take_damage(damage)
			create_impact_effect()
			queue_free()

func _on_area_entered(area):
	# Hit something, destroy
	create_impact_effect()
	queue_free()

func create_impact_effect():
	# TODO: Add impact particles/sound
	pass
