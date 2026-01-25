extends Control
# ResourceBar.gd - Displays player resources with tooltips and income indicators

class_name ResourceBar

# Resource definitions
var resources = {
	"gold": {"amount": 100, "income": 5, "icon": "💰", "description": "Gold - Used for buildings and units"},
	"wood": {"amount": 50, "income": 3, "icon": "🪵", "description": "Wood - Used for construction"},
	"food": {"amount": 30, "income": 2, "icon": "🌾", "description": "Food - Needed to support units"},
	"stone": {"amount": 20, "income": 1, "icon": "🪨", "description": "Stone - Required for advanced buildings"}
}

# UI nodes
var resource_labels = {}
var income_labels = {}
var tooltips = {}

# Signals
signal resource_changed(resource_type, new_amount)
signal insufficient_resources(resource_type, required, current)

func _ready():
	# Setup UI with tooltips
	_create_resource_ui()
	_start_income_timer()
	
	# Show tutorial on first launch
	if _is_first_time():
		_show_tutorial()

func _create_resource_ui():
	# Create horizontal container for resources
	var hbox = HBoxContainer.new()
	hbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	add_child(hbox)
	
	for resource_type in resources.keys():
		var resource_data = resources[resource_type]
		
		# Create panel for each resource
		var panel = PanelContainer.new()
		panel.custom_minimum_size = Vector2(120, 60)
		hbox.add_child(panel)
		
		var vbox = VBoxContainer.new()
		panel.add_child(vbox)
		
		# Resource icon and amount
		var amount_label = Label.new()
		amount_label.text = "%s %d" % [resource_data.icon, resource_data.amount]
		amount_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		vbox.add_child(amount_label)
		resource_labels[resource_type] = amount_label
		
		# Income indicator
		var income_label = Label.new()
		income_label.text = "+%d/sec" % resource_data.income
		income_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		income_label.modulate = Color.GREEN
		vbox.add_child(income_label)
		income_labels[resource_type] = income_label
		
		# Tooltip on hover
		var tooltip = _create_tooltip(resource_type, resource_data.description)
		panel.tooltip_text = "%s\n%s\nIncome: +%d per second" % [
			resource_type.capitalize(),
			resource_data.description,
			resource_data.income
		]

func _start_income_timer():
	# Add resources every second based on income
	var timer = Timer.new()
	timer.wait_time = 1.0
	timer.autostart = true
	timer.timeout.connect(_on_income_tick)
	add_child(timer)

func _on_income_tick():
	for resource_type in resources.keys():
		var income = resources[resource_type].income
		add_resource(resource_type, income)

func add_resource(resource_type: String, amount: int):
	"""Add resources and update UI"""
	if resource_type in resources:
		resources[resource_type].amount += amount
		_update_resource_display(resource_type)
		resource_changed.emit(resource_type, resources[resource_type].amount)

func can_afford(costs: Dictionary) -> bool:
	"""Check if player can afford the given costs"""
	for resource_type in costs.keys():
		if resource_type in resources:
			if resources[resource_type].amount < costs[resource_type]:
				return false
	return true

func spend_resources(costs: Dictionary) -> bool:
	"""
	Attempt to spend resources. Returns true if successful.
	Shows visual feedback if cannot afford.
	"""
	if not can_afford(costs):
		_show_insufficient_resources_feedback(costs)
		return false
	
	# Deduct resources
	for resource_type in costs.keys():
		if resource_type in resources:
			resources[resource_type].amount -= costs[resource_type]
			_update_resource_display(resource_type)
			resource_changed.emit(resource_type, resources[resource_type].amount)
	
	return true

func get_resource_amount(resource_type: String) -> int:
	"""Get current amount of a resource"""
	if resource_type in resources:
		return resources[resource_type].amount
	return 0

func _update_resource_display(resource_type: String):
	"""Update the display for a specific resource"""
	if resource_type in resource_labels:
		var resource_data = resources[resource_type]
		resource_labels[resource_type].text = "%s %d" % [
			resource_data.icon,
			resource_data.amount
		]

func _show_insufficient_resources_feedback(costs: Dictionary):
	"""Visual feedback when player cannot afford something"""
	for resource_type in costs.keys():
		if resource_type in resources:
			var current = resources[resource_type].amount
			var required = costs[resource_type]
			
			if current < required:
				# Flash the resource label red
				if resource_type in resource_labels:
					var label = resource_labels[resource_type]
					var tween = create_tween()
					tween.tween_property(label, "modulate", Color.RED, 0.2)
					tween.tween_property(label, "modulate", Color.WHITE, 0.2)
					tween.set_loops(3)
				
				insufficient_resources.emit(resource_type, required, current)

func _create_tooltip(resource_type: String, description: String) -> String:
	"""Create detailed tooltip text"""
	return """[b]%s[/b]
%s

[color=green]Income: +%d per second[/color]

[i]Tip: Build more gatherers to increase income![/i]""" % [
		resource_type.capitalize(),
		description,
		resources[resource_type].income
	]

func _is_first_time() -> bool:
	"""Check if this is the player's first time"""
	# Check for a save file or config
	return not FileAccess.file_exists("user://rts_tutorial_complete.dat")

func _show_tutorial():
	"""Show tutorial overlay explaining resources"""
	var tutorial = AcceptDialog.new()
	tutorial.dialog_text = """[center][b]Welcome to Resource Management![/b][/center]

[b]Resources:[/b]
• 💰 Gold - Build structures and train units
• 🪵 Wood - Construct buildings
• 🌾 Food - Support your army
• 🪨 Stone - Advanced buildings

[b]Tips:[/b]
• Resources generate automatically over time
• Check the [color=green]+X/sec[/color] indicator
• Hover over resources for details
• Red flashing means you can't afford something
• Build gatherers to increase income!"""
	
	tutorial.ok_button_text = "Got it!"
	add_child(tutorial)
	tutorial.popup_centered()
	tutorial.confirmed.connect(_on_tutorial_complete)

func _on_tutorial_complete():
	"""Mark tutorial as complete"""
	var file = FileAccess.open("user://rts_tutorial_complete.dat", FileAccess.WRITE)
	if file:
		file.store_string("1")
		file.close()
