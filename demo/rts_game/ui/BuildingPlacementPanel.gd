extends Control
# BuildingPlacementPanel.gd - UI for placing buildings with resource cost display

class_name BuildingPlacementPanel

# Reference to resource bar
@export var resource_bar: ResourceBar

# Building types available for construction
var building_types = {
	"gold_mine": {
		"name": "Gold Mine",
		"icon": "⛏️",
		"description": "Generates gold",
		"scene": "res://demo/rts_game/buildings/GoldMine.gd",
		"costs": {"gold": 150, "wood": 100, "stone": 50}
	},
	"barracks": {
		"name": "Barracks",
		"icon": "🏰",
		"description": "Train military units",
		"scene": "res://demo/rts_game/buildings/Barracks.gd",
		"costs": {"gold": 200, "wood": 150, "stone": 100}
	},
	"farm": {
		"name": "Farm",
		"icon": "🌾",
		"description": "Produces food",
		"scene": "res://demo/rts_game/buildings/Farm.gd",
		"costs": {"gold": 80, "wood": 60}
	},
	"lumbermill": {
		"name": "Lumber Mill",
		"icon": "🪓",
		"description": "Generates wood",
		"scene": null,
		"costs": {"gold": 100, "stone": 30}
	},
	"quarry": {
		"name": "Quarry",
		"icon": "🪨",
		"description": "Produces stone",
		"scene": null,
		"costs": {"gold": 120, "wood": 80}
	}
}

# UI elements
var building_buttons = {}
var placement_mode: String = ""

# Signals
signal building_placement_started(building_type)
signal building_placed(building_type, position)
signal placement_cancelled()

func _ready():
	_create_building_ui()
	set_process(true)

func _process(_delta):
	_update_affordability_indicators()

func _create_building_ui():
	# Main container
	var vbox = VBoxContainer.new()
	vbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	add_child(vbox)
	
	# Title
	var title = Label.new()
	title.text = "Build Structures"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 20)
	vbox.add_child(title)
	
	# Separator
	var separator = HSeparator.new()
	vbox.add_child(separator)
	
	# Building buttons grid
	var grid = GridContainer.new()
	grid.columns = 2
	vbox.add_child(grid)
	
	for building_type in building_types.keys():
		var building_data = building_types[building_type]
		var button = _create_building_button(building_type, building_data)
		grid.add_child(button)
		building_buttons[building_type] = button
	
	# Info section
	var info_separator = HSeparator.new()
	vbox.add_child(info_separator)
	
	var info_label = Label.new()
	info_label.text = "[i]Click a building to start placement[/i]"
	info_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	info_label.name = "InfoLabel"
	vbox.add_child(info_label)

func _create_building_button(building_type: String, building_data: Dictionary) -> Button:
	"""Create a button for a building with cost display"""
	var button = Button.new()
	button.custom_minimum_size = Vector2(180, 100)
	
	# Format costs
	var costs_text = _format_costs(building_data.costs)
	
	button.text = "%s\n%s\n%s\nCost: %s" % [
		building_data.icon,
		building_data.name,
		building_data.description,
		costs_text
	]
	
	# Detailed tooltip
	button.tooltip_text = """[b]%s %s[/b]
%s

[b]Costs:[/b]
%s

[i]Click to place this building[/i]""" % [
		building_data.icon,
		building_data.name,
		building_data.description,
		_format_costs_detailed(building_data.costs)
	]
	
	button.pressed.connect(func(): _on_building_button_pressed(building_type))
	
	return button

func _format_costs(costs: Dictionary) -> String:
	"""Format costs for button display"""
	var parts = []
	for resource_type in costs.keys():
		var icon = _get_resource_icon(resource_type)
		parts.append("%s%d" % [icon, costs[resource_type]])
	return " ".join(parts)

func _format_costs_detailed(costs: Dictionary) -> String:
	"""Format costs with detailed breakdown"""
	var parts = []
	for resource_type in costs.keys():
		var icon = _get_resource_icon(resource_type)
		parts.append("  %s %s: %d" % [icon, resource_type.capitalize(), costs[resource_type]])
	return "\n".join(parts)

func _get_resource_icon(resource_type: String) -> String:
	"""Get icon for resource type"""
	var icons = {
		"gold": "💰",
		"wood": "🪵",
		"food": "🌾",
		"stone": "🪨"
	}
	return icons.get(resource_type, "❓")

func _on_building_button_pressed(building_type: String):
	"""Handle building button press"""
	if not resource_bar:
		push_error("ResourceBar not connected to BuildingPlacementPanel!")
		return
	
	var building_data = building_types[building_type]
	
	# Check if can afford
	if not resource_bar.can_afford(building_data.costs):
		_show_cannot_afford_message(building_type, building_data.costs)
		return
	
	# Enter placement mode
	_start_placement_mode(building_type)

func _start_placement_mode(building_type: String):
	"""Enter building placement mode"""
	placement_mode = building_type
	building_placement_started.emit(building_type)
	
	# Update info label
	var info_label = get_node_or_null("VBoxContainer/InfoLabel")
	if info_label:
		info_label.text = "[color=yellow]Click on the map to place %s[/color]" % building_types[building_type].name
	
	print("Placement mode: %s - Click on the map to place" % building_types[building_type].name)

func _input(event):
	"""Handle input for building placement"""
	if placement_mode == "":
		return
	
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT:
			# Place building
			_place_building(event.position)
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			# Cancel placement
			_cancel_placement()

func _place_building(pos: Vector2):
	"""Place a building at the specified position"""
	if placement_mode == "":
		return
	
	var building_data = building_types[placement_mode]
	
	# Spend resources
	if resource_bar.spend_resources(building_data.costs):
		building_placed.emit(placement_mode, pos)
		print("Building placed: %s at %s" % [building_data.name, pos])
		
		# Exit placement mode
		_cancel_placement()
	else:
		_show_cannot_afford_message(placement_mode, building_data.costs)

func _cancel_placement():
	"""Cancel building placement mode"""
	placement_mode = ""
	placement_cancelled.emit()
	
	# Update info label
	var info_label = get_node_or_null("VBoxContainer/InfoLabel")
	if info_label:
		info_label.text = "[i]Click a building to start placement[/i]"

func _update_affordability_indicators():
	"""Update button appearance based on affordability"""
	if not resource_bar:
		return
	
	for building_type in building_buttons.keys():
		var button = building_buttons[building_type]
		var building_data = building_types[building_type]
		
		var can_afford = resource_bar.can_afford(building_data.costs)
		
		# Visual feedback
		if can_afford:
			button.modulate = Color.WHITE
			button.disabled = false
		else:
			button.modulate = Color(0.5, 0.5, 0.5, 0.7)  # Grayed out
			button.disabled = false  # Keep enabled for tooltip/messages
		
		# Update button text with affordability info
		var costs_text = _format_costs_with_affordability(building_data.costs)
		button.text = "%s\n%s\n%s\n%s" % [
			building_data.icon,
			building_data.name,
			building_data.description,
			costs_text
		]

func _format_costs_with_affordability(costs: Dictionary) -> String:
	"""Format costs showing what's affordable"""
	if not resource_bar:
		return "Cost: " + _format_costs(costs)
	
	var all_affordable = true
	var parts = []
	
	for resource_type in costs.keys():
		var icon = _get_resource_icon(resource_type)
		var cost = costs[resource_type]
		var current = resource_bar.get_resource_amount(resource_type)
		
		if current >= cost:
			parts.append("%s%d" % [icon, cost])
		else:
			parts.append("%s%d (-%d)" % [icon, cost, cost - current])
			all_affordable = false
	
	var prefix = "Cost: " if all_affordable else "Need: "
	return prefix + " ".join(parts)

func _show_cannot_afford_message(building_type: String, costs: Dictionary):
	"""Show message when player cannot afford a building"""
	var missing = []
	
	for resource_type in costs.keys():
		var required = costs[resource_type]
		var current = resource_bar.get_resource_amount(resource_type)
		
		if current < required:
			missing.append("%s %s: need %d more" % [
				_get_resource_icon(resource_type),
				resource_type.capitalize(),
				required - current
			])
	
	var building_data = building_types[building_type]
	
	var dialog = AcceptDialog.new()
	dialog.dialog_text = """[center][b]Insufficient Resources![/b][/center]

To build %s %s, you need:

%s

[i]Wait for resource income or build resource-generating buildings![/i]""" % [
		building_data.icon,
		building_data.name,
		"\n".join(missing)
	]
	
	add_child(dialog)
	dialog.popup_centered()
	dialog.confirmed.connect(dialog.queue_free)
