extends Node2D

## Example scene demonstrating the Wall Building System
## This shows how to integrate WallSystem and WallBuildPanel

# References
@onready var wall_system: WallSystem = $WallSystem
@onready var ui_panel: WallBuildPanel = $CanvasLayer/UI/WallBuildPanel

# Game state
var player_resources: int = 500

func _ready():
	# Connect wall system to UI panel
	wall_system.set_ui_panel(ui_panel)
	
	# Connect wall system signals
	wall_system.wall_placed.connect(_on_wall_placed)
	wall_system.wall_cancelled.connect(_on_wall_cancelled)
	wall_system.placement_mode_changed.connect(_on_placement_mode_changed)
	
	# Connect UI panel signals
	ui_panel.build_requested.connect(_on_build_requested)
	ui_panel.cancel_requested.connect(_on_cancel_requested)
	
	# Set initial resources
	ui_panel.set_player_resources(player_resources)
	
	print("Wall System Example loaded!")
	print("Click 'Build Wall' to start")

func _on_build_requested():
	"""Handle build button press from UI"""
	var cost = wall_system.get_placement_cost()
	
	if wall_system.can_afford_wall(player_resources):
		wall_system.start_placement()
		print("Entering wall placement mode")
	else:
		ui_panel.show_error("Not enough resources! Need %d, have %d" % [cost, player_resources])
		print("Cannot afford wall: need %d, have %d" % [cost, player_resources])

func _on_cancel_requested():
	"""Handle cancel button press from UI"""
	wall_system.cancel_placement()
	print("Wall placement cancelled from UI")

func _on_wall_placed(position: Vector2):
	"""Handle successful wall placement"""
	var cost = wall_system.get_placement_cost()
	
	# Deduct resources
	player_resources -= cost
	ui_panel.set_player_resources(player_resources)
	
	# Show success message
	ui_panel.show_success("Wall placed successfully!")
	
	print("Wall placed at ", position, " - Resources remaining: ", player_resources)

func _on_wall_cancelled():
	"""Handle wall placement cancellation"""
	ui_panel.show_status("Wall placement cancelled")
	print("Wall placement cancelled")

func _on_placement_mode_changed(active: bool):
	"""Handle placement mode state change"""
	ui_panel.set_building_mode(active)
	
	if active:
		print("Placement mode: ACTIVE")
	else:
		print("Placement mode: INACTIVE")

# Debug: Add resources with a key press
func _input(event: InputEvent):
	if event is InputEventKey and event.pressed:
		# Press 'R' to add resources
		if event.keycode == KEY_R:
			player_resources += 100
			ui_panel.set_player_resources(player_resources)
			print("Added 100 resources. Total: ", player_resources)
