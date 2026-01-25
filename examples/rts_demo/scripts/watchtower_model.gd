extends MeshInstance3D

## Simple watchtower 3D model created programmatically

func _ready():
	_create_tower_mesh()

func _create_tower_mesh():
	"""Create a simple tower structure"""
	var arr_mesh = ArrayMesh.new()
	
	# Base platform
	_add_box_to_mesh(arr_mesh, Vector3.ZERO, Vector3(3, 0.5, 3), Color(0.6, 0.5, 0.4))
	
	# Tower body
	_add_box_to_mesh(arr_mesh, Vector3(0, 1.5, 0), Vector3(1.5, 3, 1.5), Color(0.7, 0.6, 0.5))
	
	# Top platform
	_add_box_to_mesh(arr_mesh, Vector3(0, 3.5, 0), Vector3(2, 0.3, 2), Color(0.5, 0.4, 0.3))
	
	# Battlements (4 corners)
	for corner in [Vector3(0.8, 4, 0.8), Vector3(-0.8, 4, 0.8), 
				   Vector3(0.8, 4, -0.8), Vector3(-0.8, 4, -0.8)]:
		_add_box_to_mesh(arr_mesh, corner, Vector3(0.4, 0.6, 0.4), Color(0.6, 0.5, 0.4))
	
	mesh = arr_mesh
	
	# Add collision for mouse detection
	var collision = StaticBody3D.new()
	var collision_shape = CollisionShape3D.new()
	var box_shape = BoxShape3D.new()
	box_shape.size = Vector3(2, 5, 2)
	collision_shape.shape = box_shape
	collision_shape.position = Vector3(0, 2, 0)
	collision.add_child(collision_shape)
	add_child(collision)

func _add_box_to_mesh(arr_mesh: ArrayMesh, pos: Vector3, size: Vector3, color: Color):
	"""Add a box to the array mesh"""
	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	
	var verts = PackedVector3Array()
	var normals = PackedVector3Array()
	var colors = PackedColorArray()
	
	var hs = size / 2  # Half size
	
	# Define 8 vertices of a box
	var v = [
		pos + Vector3(-hs.x, -hs.y, -hs.z),
		pos + Vector3(hs.x, -hs.y, -hs.z),
		pos + Vector3(hs.x, -hs.y, hs.z),
		pos + Vector3(-hs.x, -hs.y, hs.z),
		pos + Vector3(-hs.x, hs.y, -hs.z),
		pos + Vector3(hs.x, hs.y, -hs.z),
		pos + Vector3(hs.x, hs.y, hs.z),
		pos + Vector3(-hs.x, hs.y, hs.z),
	]
	
	# Define faces (6 faces, 2 triangles each)
	var faces = [
		# Bottom
		[v[0], v[1], v[2], Vector3(0, -1, 0)],
		[v[0], v[2], v[3], Vector3(0, -1, 0)],
		# Top
		[v[4], v[6], v[5], Vector3(0, 1, 0)],
		[v[4], v[7], v[6], Vector3(0, 1, 0)],
		# Front
		[v[3], v[2], v[6], Vector3(0, 0, 1)],
		[v[3], v[6], v[7], Vector3(0, 0, 1)],
		# Back
		[v[0], v[5], v[1], Vector3(0, 0, -1)],
		[v[0], v[4], v[5], Vector3(0, 0, -1)],
		# Left
		[v[0], v[3], v[7], Vector3(-1, 0, 0)],
		[v[0], v[7], v[4], Vector3(-1, 0, 0)],
		# Right
		[v[1], v[6], v[2], Vector3(1, 0, 0)],
		[v[1], v[5], v[6], Vector3(1, 0, 0)],
	]
	
	for face in faces:
		for i in range(3):
			verts.append(face[i])
			normals.append(face[3])
			colors.append(color)
	
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_COLOR] = colors
	
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
