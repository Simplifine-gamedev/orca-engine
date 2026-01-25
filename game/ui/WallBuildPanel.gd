extends Control
class_name WallBuildPanel

## Wall Building UI Panel with improved UX
## Features:
## - Cost preview before confirming
## - Tutorial tooltip on first build
## - Clear visual feedback
## - Status messages

signal build_requested
signal cancel_requested

# UI References (assigned from scene tree)
@onready var cost_label: Label = $CostPanel/CostLabel
@onready var cost_panel: PanelContainer = $CostPanel
@onready var tutorial_popup: PopupPanel = $TutorialPopup
@onready var tutorial_label: Label = $TutorialPopup/TutorialLabel
@onready var status_label: Label = $StatusLabel
@onready var build_button: Button = $BuildButton
@onready var cancel_button: Button = $CancelButton
@onready var resources_label: Label = $ResourcesLabel

# State
var current_cost: int = 0
var player_resources: int = 0
var is_building: bool = false

func _ready():
	# Connect button signals
	if build_button:
		build_button.pressed.connect(_on_build_pressed)
	if cancel_button:
		cancel_button.pressed.connect(_on_cancel_pressed)
	
	# Initial state
	hide_cost_preview()
	hide_tutorial()
	update_ui_state()

func show_cost_preview(cost: int):
	"""Show cost preview during wall placement"""
	current_cost = cost
	
	if cost_label:
		cost_label.text = "Cost: %d resources" % cost
	
	if cost_panel:
		cost_panel.visible = true
		
		# Color code based on affordability
		if player_resources >= cost:
			cost_panel.modulate = Color(0.8, 1.0, 0.8)  # Light green - can afford
		else:
			cost_panel.modulate = Color(1.0, 0.8, 0.8)  # Light red - cannot afford
	
	update_status_message("Place wall or right-click to cancel")

func hide_cost_preview():
	"""Hide cost preview when not placing"""
	if cost_panel:
		cost_panel.visible = false
	current_cost = 0

func show_tutorial(title: String, message: String):
	"""Show tutorial popup on first wall build"""
	if tutorial_label:
		tutorial_label.text = "[center][b]%s[/b][/center]\n\n%s" % [title, message]
	
	if tutorial_popup:
		tutorial_popup.popup_centered()
		
		# Auto-hide after delay
		await get_tree().create_timer(5.0).timeout
		hide_tutorial()

func hide_tutorial():
	"""Hide tutorial popup"""
	if tutorial_popup:
		tutorial_popup.hide()

func update_status_message(message: String):
	"""Update status message for user feedback"""
	if status_label:
		status_label.text = message
		status_label.modulate = Color.WHITE

func show_error(message: String):
	"""Show error message with red highlight"""
	if status_label:
		status_label.text = message
		status_label.modulate = Color.RED
		
		# Flash animation
		var tween = create_tween()
		tween.tween_property(status_label, "modulate:a", 0.0, 0.5)
		tween.tween_property(status_label, "modulate:a", 1.0, 0.5)
		tween.set_loops(3)

func show_success(message: String):
	"""Show success message with green highlight"""
	if status_label:
		status_label.text = message
		status_label.modulate = Color.GREEN
		
		# Fade out
		await get_tree().create_timer(1.0).timeout
		var tween = create_tween()
		tween.tween_property(status_label, "modulate:a", 0.0, 1.0)

func set_player_resources(resources: int):
	"""Update player's available resources"""
	player_resources = resources
	
	if resources_label:
		resources_label.text = "Resources: %d" % resources
	
	# Update cost preview color if visible
	if cost_panel and cost_panel.visible:
		show_cost_preview(current_cost)

func set_building_mode(active: bool):
	"""Update UI for building mode state"""
	is_building = active
	update_ui_state()
	
	if active:
		update_status_message("Building mode active - Right-click to cancel")
	else:
		update_status_message("Click 'Build Wall' to start")

func update_ui_state():
	"""Update button states based on current state"""
	if build_button:
		build_button.disabled = is_building
		build_button.text = "Building..." if is_building else "Build Wall"
	
	if cancel_button:
		cancel_button.visible = is_building

func _on_build_pressed():
	"""Handle build button press"""
	if player_resources >= current_cost or current_cost == 0:
		build_requested.emit()
		set_building_mode(true)
	else:
		show_error("Not enough resources!")

func _on_cancel_pressed():
	"""Handle cancel button press"""
	cancel_requested.emit()
	set_building_mode(false)

func create_standalone_panel() -> Control:
	"""Create a standalone panel with all UI elements (for testing/standalone use)"""
	var panel = PanelContainer.new()
	panel.name = "WallBuildPanel"
	
	var vbox = VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 10)
	panel.add_child(vbox)
	
	# Title
	var title = Label.new()
	title.text = "Wall Building"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 20)
	vbox.add_child(title)
	
	# Resources display
	var res_label = Label.new()
	res_label.name = "ResourcesLabel"
	res_label.text = "Resources: 0"
	vbox.add_child(res_label)
	
	# Cost panel
	var cost_container = PanelContainer.new()
	cost_container.name = "CostPanel"
	cost_container.visible = false
	var cost_lbl = Label.new()
	cost_lbl.name = "CostLabel"
	cost_lbl.text = "Cost: 0"
	cost_container.add_child(cost_lbl)
	vbox.add_child(cost_container)
	
	# Status message
	var status = Label.new()
	status.name = "StatusLabel"
	status.text = "Ready to build"
	status.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(status)
	
	# Buttons
	var button_container = HBoxContainer.new()
	button_container.add_theme_constant_override("separation", 10)
	
	var build_btn = Button.new()
	build_btn.name = "BuildButton"
	build_btn.text = "Build Wall"
	button_container.add_child(build_btn)
	
	var cancel_btn = Button.new()
	cancel_btn.name = "CancelButton"
	cancel_btn.text = "Cancel"
	cancel_btn.visible = false
	button_container.add_child(cancel_btn)
	
	vbox.add_child(button_container)
	
	# Tutorial popup
	var tutorial = PopupPanel.new()
	tutorial.name = "TutorialPopup"
	tutorial.size = Vector2(300, 200)
	var tutorial_text = Label.new()
	tutorial_text.name = "TutorialLabel"
	tutorial_text.text = "Tutorial"
	tutorial_text.autowrap_mode = TextServer.AUTOWRAP_WORD
	tutorial.add_child(tutorial_text)
	panel.add_child(tutorial)
	
	return panel

# Keyboard shortcuts
func _input(event: InputEvent):
	if event.is_action_pressed("ui_cancel") and is_building:
		_on_cancel_pressed()

func get_help_text() -> String:
	"""Return help text for the wall building system"""
	return """
	Wall Building Guide:
	
	1. Click 'Build Wall' to enter placement mode
	2. Move mouse to desired location
	3. Valid placement areas are highlighted in green
	4. Invalid areas are shown in red
	5. Cost is displayed at the top
	6. Left-click to place the wall
	7. Right-click to cancel placement
	8. ESC key also cancels placement
	
	Tips:
	- Walls snap to a grid for clean placement
	- Check your resources before building
	- Walls cannot be placed on water or other buildings
	"""
