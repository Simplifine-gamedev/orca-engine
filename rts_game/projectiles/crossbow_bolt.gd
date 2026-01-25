extends Area3D
class_name CrossbowBolt

# Crossbow bolt projectile fired by crossbowmen
# Travels faster and deals more damage with armor penetration

var direction: Vector3 = Vector3.FORWARD
var speed: float = 25.0
var damage: float = 18.0
var faction_id: int = 0
var armor_penetration: float = 5.0
var max_distance: float = 35.0
var traveled_distance: float = 0.0

@onready var mesh: MeshInstance3D = $BoltMesh
@onready var trail: GPUParticles3D = $Trail

func _ready():
	setup_visuals()
	body_entered.connect(_on_body_entered)
	area_entered.connect(_on_area_entered)

func setup_visuals():
	# Create crossbow bolt mesh if not exists
	if not mesh:
		mesh = MeshInstance3D.new()
		add_child(mesh)
		
		var bolt_mesh = CylinderMesh.new()
		bolt_mesh.height = 0.6
		bolt_mesh.top_radius = 0.025
		bolt_mesh.bottom_radius = 0.025
		mesh.mesh = bolt_mesh
		
		# Rotate to point forward
		mesh.rotation.z = deg_to_rad(90)
		
		# Dark wood color (crossbow bolt)
		var material = StandardMaterial3D.new()
		material.albedo_color = Color(0.3, 0.25, 0.2)
		mesh.set_surface_override_material(0, material)
	
	# Add bolt head (larger and more robust than arrow)
	var bolthead = MeshInstance3D.new()
	var cone_mesh = CylinderMesh.new()
	cone_mesh.height = 0.15
	cone_mesh.top_radius = 0.0
	cone_mesh.bottom_radius = 0.06
	bolthead.mesh = cone_mesh
	bolthead.position = Vector3(0.375, 0, 0)
	bolthead.rotation.z = deg_to_rad(90)
	var metal_material = StandardMaterial3D.new()
	metal_material.albedo_color = Color(0.5, 0.5, 0.5)
	metal_material.metallic = 0.9
	bolthead.set_surface_override_material(0, metal_material)
	add_child(bolthead)

func set_direction(dir: Vector3, proj_speed: float, proj_damage: float, proj_faction: int, armor_pen: float = 5.0):
	direction = dir.normalized()
	speed = proj_speed
	damage = proj_damage
	faction_id = proj_faction
	armor_penetration = armor_pen
	
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
				# Apply armor penetration
				var total_damage = damage + armor_penetration
				body.take_damage(total_damage)
				create_impact_effect()
				queue_free()
		else:
			body.take_damage(damage + armor_penetration)
			create_impact_effect()
			queue_free()

func _on_area_entered(area):
	# Hit something, destroy
	create_impact_effect()
	queue_free()

func create_impact_effect():
	# TODO: Add impact particles/sound
	# Crossbow bolts create heavier impact than arrows
	pass
