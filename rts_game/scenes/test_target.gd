extends CharacterBody3D

# Simple test target that takes damage and reports it

var health: float = 1000.0
var max_health: float = 1000.0

func _ready():
	print("Test target spawned with ", health, " HP")

func take_damage(amount: float):
	health -= amount
	print("Target took ", amount, " damage. Health: ", health, "/", max_health)
	
	if health <= 0:
		print("Target destroyed!")
		queue_free()

func get_faction() -> int:
	return 99  # Different faction from units
