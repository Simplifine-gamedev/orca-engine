extends Control
# WorkerBuildPanel.gd - UI for building and training units with cost display

class_name WorkerBuildPanel

# Reference to resource bar
@export var resource_bar: ResourceBar

# Unit definitions with costs
var units = {
	"worker": {
		"name": "Worker",
		"icon": "👷",
		"description": "Gathers resources",
		"costs": {"gold": 50, "food": 10},
		"build_time": 5.0
	},
	"soldier": {
		"name": "Soldier",
		"icon": "⚔️",
		"description": "Basic combat unit",
		"costs": {"gold": 100, "food": 25},
		"build_time": 8.0
	},
	"archer": {
		"name": "Archer",
		"icon": "🏹",
		"description": "Ranged unit",
		"costs": {"gold": 80, "wood": 30, "food": 20},
		"build_time": 10.0
	},
	"cavalry": {
		"name": "Cavalry",
		"icon": "🐎",
		"description": "Fast mounted unit",
		"costs": {"gold": 150, "food": 40},
		"build_time": 15.0
	}
}

# Build queue
var build_queue = []
var current_building = null

# UI elements
var unit_buttons = {}

# Signals
signal unit_training_started(unit_type)
signal unit_training_complete(unit_type)
signal build_queue_updated(queue_size)

func _ready():
	_create_build_ui()
	
	# Update affordability every frame
	set_process(true)

func _process(_delta):
	_update_affordability_indicators()

func _create_build_ui():
	# Main container
	var vbox = VBoxContainer.new()
	vbox.size_flags_vertical = Control.SIZE_EXPAND_FILL
	add_child(vbox)
	
	# Title
	var title = Label.new()
	title.text = "Train Units"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 20)
	vbox.add_child(title)
	
	# Separator
	var separator = HSeparator.new()
	vbox.add_child(separator)
	
	# Unit buttons
	for unit_type in units.keys():
		var unit_data = units[unit_type]
		var button = _create_unit_button(unit_type, unit_data)
		vbox.add_child(button)
		unit_buttons[unit_type] = button
	
	# Build queue display
	var queue_separator = HSeparator.new()
	vbox.add_child(queue_separator)
	
	var queue_label = Label.new()
	queue_label.text = "Build Queue"
	queue_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(queue_label)
	
	var queue_container = VBoxContainer.new()
	queue_container.name = "QueueContainer"
	queue_container.custom_minimum_size = Vector2(0, 100)
	vbox.add_child(queue_container)

func _create_unit_button(unit_type: String, unit_data: Dictionary) -> Button:
	"""Create a button for training a unit with cost display"""
	var button = Button.new()
	button.custom_minimum_size = Vector2(0, 80)
	
	# Create rich text for button
	var costs_text = _format_costs(unit_data.costs)
	button.text = "%s %s\n%s\n%s" % [
		unit_data.icon,
		unit_data.name,
		unit_data.description,
		costs_text
	]
	
	# Tooltip with detailed info
	button.tooltip_text = """[b]%s %s[/b]
%s

[b]Costs:[/b]
%s

[b]Build Time:[/b] %.1f seconds""" % [
		unit_data.icon,
		unit_data.name,
		unit_data.description,
		_format_costs_detailed(unit_data.costs),
		unit_data.build_time
	]
	
	button.pressed.connect(func(): _on_unit_button_pressed(unit_type))
	
	return button

func _format_costs(costs: Dictionary) -> String:
	"""Format costs for display on button"""
	var parts = []
	for resource_type in costs.keys():
		var icon = _get_resource_icon(resource_type)
		parts.append("%s %d" % [icon, costs[resource_type]])
	return " | ".join(parts)

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

func _on_unit_button_pressed(unit_type: String):
	"""Handle unit training button press"""
	if not resource_bar:
		push_error("ResourceBar not connected to WorkerBuildPanel!")
		return
	
	var unit_data = units[unit_type]
	
	# Check if can afford
	if not resource_bar.can_afford(unit_data.costs):
		_show_cannot_afford_message(unit_type, unit_data.costs)
		return
	
	# Spend resources
	if resource_bar.spend_resources(unit_data.costs):
		_add_to_build_queue(unit_type)

func _add_to_build_queue(unit_type: String):
	"""Add unit to build queue"""
	build_queue.append(unit_type)
	_update_queue_display()
	build_queue_updated.emit(build_queue.size())
	
	# Start building if not already building
	if current_building == null:
		_start_next_build()

func _start_next_build():
	"""Start building next unit in queue"""
	if build_queue.is_empty():
		current_building = null
		return
	
	current_building = build_queue.pop_front()
	var unit_data = units[current_building]
	
	unit_training_started.emit(current_building)
	
	# Create timer for build time
	var timer = Timer.new()
	timer.wait_time = unit_data.build_time
	timer.one_shot = true
	timer.timeout.connect(_on_build_complete)
	add_child(timer)
	timer.start()
	
	_update_queue_display()

func _on_build_complete():
	"""Called when a unit finishes training"""
	if current_building:
		unit_training_complete.emit(current_building)
		print("Unit trained: %s %s" % [
			units[current_building].icon,
			units[current_building].name
		])
	
	# Start next build
	_start_next_build()

func _update_queue_display():
	"""Update the visual display of the build queue"""
	var queue_container = get_node_or_null("VBoxContainer/QueueContainer")
	if not queue_container:
		return
	
	# Clear existing
	for child in queue_container.get_children():
		child.queue_free()
	
	# Show current building
	if current_building:
		var label = Label.new()
		label.text = "🔨 Building: %s %s" % [
			units[current_building].icon,
			units[current_building].name
		]
		label.modulate = Color.YELLOW
		queue_container.add_child(label)
	
	# Show queue
	for i in range(min(3, build_queue.size())):
		var unit_type = build_queue[i]
		var label = Label.new()
		label.text = "%d. %s %s" % [
			i + 1,
			units[unit_type].icon,
			units[unit_type].name
		]
		queue_container.add_child(label)
	
	# Show more indicator
	if build_queue.size() > 3:
		var more = Label.new()
		more.text = "... +%d more" % (build_queue.size() - 3)
		queue_container.add_child(more)

func _update_affordability_indicators():
	"""Update button appearance based on affordability"""
	if not resource_bar:
		return
	
	for unit_type in unit_buttons.keys():
		var button = unit_buttons[unit_type]
		var unit_data = units[unit_type]
		
		var can_afford = resource_bar.can_afford(unit_data.costs)
		
		# Visual feedback
		if can_afford:
			button.modulate = Color.WHITE
			button.disabled = false
		else:
			button.modulate = Color(0.5, 0.5, 0.5, 0.7)  # Grayed out
			button.disabled = false  # Keep enabled for tooltip
		
		# Update button text with colored costs
		var costs_text = _format_costs_with_affordability(unit_data.costs)
		button.text = "%s %s\n%s\n%s" % [
			unit_data.icon,
			unit_data.name,
			unit_data.description,
			costs_text
		]

func _format_costs_with_affordability(costs: Dictionary) -> String:
	"""Format costs showing red for unaffordable resources"""
	if not resource_bar:
		return _format_costs(costs)
	
	var parts = []
	for resource_type in costs.keys():
		var icon = _get_resource_icon(resource_type)
		var cost = costs[resource_type]
		var current = resource_bar.get_resource_amount(resource_type)
		
		# Simple text format (rich text not fully supported in Button.text)
		if current >= cost:
			parts.append("%s %d" % [icon, cost])
		else:
			parts.append("%s %d (need %d more)" % [icon, cost, cost - current])
	
	return " | ".join(parts)

func _show_cannot_afford_message(unit_type: String, costs: Dictionary):
	"""Show message when player cannot afford a unit"""
	var missing = []
	
	for resource_type in costs.keys():
		var required = costs[resource_type]
		var current = resource_bar.get_resource_amount(resource_type)
		
		if current < required:
			missing.append("%s %s (need %d more)" % [
				_get_resource_icon(resource_type),
				resource_type.capitalize(),
				required - current
			])
	
	var dialog = AcceptDialog.new()
	dialog.dialog_text = """[b]Insufficient Resources![/b]

To train %s %s, you need:

%s

[i]Wait for resource income or build more gatherers![/i]""" % [
		units[unit_type].icon,
		units[unit_type].name,
		"\n".join(missing)
	]
	
	add_child(dialog)
	dialog.popup_centered()
	dialog.confirmed.connect(dialog.queue_free)
