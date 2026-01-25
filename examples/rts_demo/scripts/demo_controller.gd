extends Node3D

## Demo controller for watchtower example
## Handles demo-specific logic and interactions

var watchtowers: Array = []
var current_player_team: int = 1

func _ready():
	# Get all watchtowers
	var watchtower_container = $Watchtowers
	if watchtower_container:
		for child in watchtower_container.get_children():
			if child is ControlPoint:
				watchtowers.append(child)
	
	print("RTS Watchtower Demo loaded!")
	print("Found ", watchtowers.size(), " watchtowers")

func _input(event):
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_ESCAPE:
				get_tree().quit()
			KEY_SPACE:
				_toggle_team_control()
			KEY_R:
				_reset_watchtowers()
			KEY_1, KEY_2, KEY_3, KEY_4, KEY_5:
				var index = event.keycode - KEY_1
				if index < watchtowers.size():
					_conquer_watchtower(watchtowers[index])

func _toggle_team_control():
	"""Cycle through team control for all watchtowers"""
	for tower in watchtowers:
		var next_team = (tower.team + 1) % 3
		tower.conquer(next_team)
	print("Toggled all watchtowers - New team states updated")

func _reset_watchtowers():
	"""Reset all watchtowers to neutral"""
	for tower in watchtowers:
		tower.conquer(0)
	print("All watchtowers reset to neutral")

func _conquer_watchtower(tower: ControlPoint):
	"""Conquer a specific watchtower for the player"""
	if tower:
		tower.conquer(current_player_team)
		print("Conquered watchtower for team ", current_player_team)

func _process(_delta):
	# Check for mouse clicks on watchtowers
	if Input.is_action_just_pressed("ui_select"):  # Usually mapped to Enter/Space
		pass  # Handle additional input if needed
