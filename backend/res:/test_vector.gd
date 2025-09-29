extends RefCounted
class_name TestVector

# Function to add two Vector2 objects
static func add_vector2(vector_a: Vector2, vector_b: Vector2) -> Vector2:
	return vector_a + vector_b

# Function to add two Vector3 objects
static func add_vector3(vector_a: Vector3, vector_b: Vector3) -> Vector3:
	return vector_a + vector_b

# Function to add two Vector2 objects and return unit vector (normalized)
static func add_vectors_to_unit_vector2(vector_a: Vector2, vector_b: Vector2) -> Vector2:
	var sum := vector_a + vector_b
	if sum.length() > 0.0:
		return sum.normalized()
	else:
		return Vector2.ZERO

# Function to add two Vector3 objects and return unit vector (normalized)
static func add_vectors_to_unit_vector3(vector_a: Vector3, vector_b: Vector3) -> Vector3:
	var sum := vector_a + vector_b
	if sum.length() > 0.0:
		return sum.normalized()
	else:
		return Vector3.ZERO

# Function to add four Vector2 objects
static func add_four_vector2(vec_a: Vector2, vec_b: Vector2, vec_c: Vector2, vec_d: Vector2) -> Vector2:
	return vec_a + vec_b + vec_c + vec_d

# Function to add four Vector3 objects
static func add_four_vector3(vec_a: Vector3, vec_b: Vector3, vec_c: Vector3, vec_d: Vector3) -> Vector3:
	return vec_a + vec_b + vec_c + vec_d

# Function to add an array of Vector2 objects (flexible for any number)
static func add_vector2_array(vectors: Array[Vector2]) -> Vector2:
	var result := Vector2.ZERO
	for vec in vectors:
		result += vec
	return result

# Function to add an array of Vector3 objects (flexible for any number)
static func add_vector3_array(vectors: Array[Vector3]) -> Vector3:
	var result := Vector3.ZERO
	for vec in vectors:
		result += vec
	return result

# Function to multiply two Vector2 objects (component-wise)
static func multiply_vector2(vector_a: Vector2, vector_b: Vector2) -> Vector2:
	return Vector2(vector_a.x * vector_b.x, vector_a.y * vector_b.y)

# Function to multiply two Vector3 objects (component-wise)
static func multiply_vector3(vector_a: Vector3, vector_b: Vector3) -> Vector3:
	return Vector3(vector_a.x * vector_b.x, vector_a.y * vector_b.y, vector_a.z * vector_b.z)

# Function to multiply Vector2 by scalar
static func multiply_vector2_scalar(vector: Vector2, scalar: float) -> Vector2:
	return vector * scalar

# Function to multiply Vector3 by scalar
static func multiply_vector3_scalar(vector: Vector3, scalar: float) -> Vector3:
	return vector * scalar

# Function to calculate dot product of two Vector2 objects
static func dot_product_vector2(vector_a: Vector2, vector_b: Vector2) -> float:
	return vector_a.dot(vector_b)

# Function to calculate dot product of two Vector3 objects
static func dot_product_vector3(vector_a: Vector3, vector_b: Vector3) -> float:
	return vector_a.dot(vector_b)

# Function to calculate cross product of two Vector3 objects
static func cross_product_vector3(vector_a: Vector3, vector_b: Vector3) -> Vector3:
	return vector_a.cross(vector_b)

# Test function to demonstrate usage
static func run_tests() -> void:
	print("=== Vector Addition Tests ===")
	
	# Test Vector2 addition
	var vec2_a := Vector2(1.5, 2.0)
	var vec2_b := Vector2(3.0, 4.5)
	var result_2d := add_vector2(vec2_a, vec2_b)
	print("Vector2 Addition: ", vec2_a, " + ", vec2_b, " = ", result_2d)
	
	# Test Vector3 addition
	var vec3_a := Vector3(1.0, 2.0, 3.0)
	var vec3_b := Vector3(4.0, 5.0, 6.0)
	var result_3d := add_vector3(vec3_a, vec3_b)
	print("Vector3 Addition: ", vec3_a, " + ", vec3_b, " = ", result_3d)
	
	# Test unit vector creation from addition
	var unit_2d := add_vectors_to_unit_vector2(vec2_a, vec2_b)
	print("Vector2 Unit Vector: ", vec2_a, " + ", vec2_b, " normalized = ", unit_2d, " (length: ", unit_2d.length(), ")")
	
	var unit_3d := add_vectors_to_unit_vector3(vec3_a, vec3_b)
	print("Vector3 Unit Vector: ", vec3_a, " + ", vec3_b, " normalized = ", unit_3d, " (length: ", unit_3d.length(), ")")
	
	# Test four Vector2 addition
	var v2_1 := Vector2(1.0, 2.0)
	var v2_2 := Vector2(3.0, 4.0)
	var v2_3 := Vector2(5.0, 6.0)
	var v2_4 := Vector2(7.0, 8.0)
	var result_four_2d := add_four_vector2(v2_1, v2_2, v2_3, v2_4)
	print("Four Vector2 Addition: ", v2_1, " + ", v2_2, " + ", v2_3, " + ", v2_4, " = ", result_four_2d)
	
	# Test four Vector3 addition
	var v3_1 := Vector3(1.0, 2.0, 3.0)
	var v3_2 := Vector3(4.0, 5.0, 6.0)
	var v3_3 := Vector3(7.0, 8.0, 9.0)
	var v3_4 := Vector3(10.0, 11.0, 12.0)
	var result_four_3d := add_four_vector3(v3_1, v3_2, v3_3, v3_4)
	print("Four Vector3 Addition: ", v3_1, " + ", v3_2, " + ", v3_3, " + ", v3_4, " = ", result_four_3d)
	
	# Test array-based addition (flexible approach)
	var vec2_array: Array[Vector2] = [Vector2(1.0, 2.0), Vector2(3.0, 4.0), Vector2(5.0, 6.0), Vector2(7.0, 8.0)]
	var array_result_2d := add_vector2_array(vec2_array)
	print("Vector2 Array Addition: ", vec2_array, " = ", array_result_2d)
	
	var vec3_array: Array[Vector3] = [Vector3(1.0, 2.0, 3.0), Vector3(4.0, 5.0, 6.0), Vector3(7.0, 8.0, 9.0), Vector3(10.0, 11.0, 12.0)]
	var array_result_3d := add_vector3_array(vec3_array)
	print("Vector3 Array Addition: ", vec3_array, " = ", array_result_3d)
	
	print("\n=== Vector Multiplication Tests ===")
	
	# Test Vector2 component-wise multiplication
	var mult_result_2d := multiply_vector2(vec2_a, vec2_b)
	print("Vector2 Multiplication: ", vec2_a, " * ", vec2_b, " = ", mult_result_2d)
	
	# Test Vector3 component-wise multiplication
	var mult_result_3d := multiply_vector3(vec3_a, vec3_b)
	print("Vector3 Multiplication: ", vec3_a, " * ", vec3_b, " = ", mult_result_3d)
	
	# Test Vector2 scalar multiplication
	var scalar_mult_2d := multiply_vector2_scalar(vec2_a, 2.5)
	print("Vector2 Scalar Multiplication: ", vec2_a, " * 2.5 = ", scalar_mult_2d)
	
	# Test Vector3 scalar multiplication
	var scalar_mult_3d := multiply_vector3_scalar(vec3_a, 3.0)
	print("Vector3 Scalar Multiplication: ", vec3_a, " * 3.0 = ", scalar_mult_3d)
	
	# Test dot product
	var dot_2d := dot_product_vector2(vec2_a, vec2_b)
	print("Vector2 Dot Product: ", vec2_a, " · ", vec2_b, " = ", dot_2d)
	
	var dot_3d := dot_product_vector3(vec3_a, vec3_b)
	print("Vector3 Dot Product: ", vec3_a, " · ", vec3_b, " = ", dot_3d)
	
	# Test cross product
	var cross_3d := cross_product_vector3(vec3_a, vec3_b)
	print("Vector3 Cross Product: ", vec3_a, " × ", vec3_b, " = ", cross_3d)
	
	print("=== Tests Complete ===")