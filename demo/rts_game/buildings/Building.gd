extends Node2D
# Building.gd - Base class for all buildings with resource cost display

class_name Building

# Building properties
@export var building_name: String = "Building"
@export var building_icon: String = "🏢"
@export var building_description: String = "A basic building"

# Resource costs
@export var gold_cost: int = 100
@export var wood_cost: int = 50
@export var stone_cost: int = 0
@export var food_cost: int = 0

# Building stats
@export var max_health: int = 1000
@export var build_time: float = 10.0
@export var provides_housing: int = 0
@export var resource_generation: Dictionary = {}  # e.g., {"gold": 5} for gold mine

var current_health: int
var is_under_construction: bool = true
var construction_progress: float = 0.0

# UI elements
var health_bar: ProgressBar
var construction_progress_bar: ProgressBar
var cost_label: Label

# Signals
signal construction_started()
signal construction_complete()
signal building_destroyed()
signal building_selected()

func _ready():
	current_health = max_health
	_create_ui_elements()
	
	if is_under_construction:
		_start_construction()

func _create_ui_elements():
	"""Create UI elements for the building"""
	# Health bar
	health_bar = ProgressBar.new()
	health_bar.max_value = max_health
	health_bar.value = current_health
	health_bar.show_percentage = false
	health_bar.position = Vector2(-50, -80)
	health_bar.size = Vector2(100, 10)
	add_child(health_bar)
	
	# Construction progress bar (initially visible if under construction)
	construction_progress_bar = ProgressBar.new()
	construction_progress_bar.max_value = 100
	construction_progress_bar.value = 0
	construction_progress_bar.show_percentage = true
	construction_progress_bar.position = Vector2(-50, -65)
	construction_progress_bar.size = Vector2(100, 15)
	construction_progress_bar.visible = is_under_construction
	add_child(construction_progress_bar)
	
	# Cost label (shown when selected or hovered)
	cost_label = Label.new()
	cost_label.text = get_cost_display()
	cost_label.position = Vector2(-60, 50)
	cost_label.visible = false
	add_child(cost_label)

func _start_construction():
	"""Start building construction"""
	is_under_construction = true
	construction_progress = 0.0
	construction_started.emit()
	
	# Make building semi-transparent during construction
	modulate = Color(1, 1, 1, 0.5)
	
	# Start construction timer
	var timer = Timer.new()
	timer.wait_time = 0.1  # Update every 0.1 seconds
	timer.timeout.connect(_on_construction_tick)
	add_child(timer)
	timer.start()

func _on_construction_tick():
	"""Update construction progress"""
	if not is_under_construction:
		return
	
	construction_progress += (100.0 / build_time) * 0.1
	
	if construction_progress_bar:
		construction_progress_bar.value = construction_progress
	
	if construction_progress >= 100.0:
		_complete_construction()

func _complete_construction():
	"""Complete building construction"""
	is_under_construction = false
	construction_progress = 100.0
	
	# Restore full opacity
	modulate = Color(1, 1, 1, 1)
	
	# Hide construction progress bar
	if construction_progress_bar:
		construction_progress_bar.visible = false
	
	construction_complete.emit()
	
	# Start resource generation if applicable
	if not resource_generation.is_empty():
		_start_resource_generation()

func _start_resource_generation():
	"""Start generating resources (for mines, farms, etc.)"""
	var timer = Timer.new()
	timer.wait_time = 1.0
	timer.autostart = true
	timer.timeout.connect(_on_resource_generation_tick)
	add_child(timer)

func _on_resource_generation_tick():
	"""Generate resources each second"""
	if is_under_construction:
		return
	
	# Find resource bar in the scene tree and add resources
	var resource_bar = _find_resource_bar()
	if resource_bar:
		for resource_type in resource_generation.keys():
			var amount = resource_generation[resource_type]
			resource_bar.add_resource(resource_type, amount)

func _find_resource_bar() -> ResourceBar:
	"""Find the ResourceBar in the scene tree"""
	# Look for ResourceBar in the root node
	var root = get_tree().root
	for child in root.get_children():
		var resource_bar = _find_resource_bar_recursive(child)
		if resource_bar:
			return resource_bar
	return null

func _find_resource_bar_recursive(node: Node) -> ResourceBar:
	"""Recursively search for ResourceBar"""
	if node is ResourceBar:
		return node
	
	for child in node.get_children():
		var result = _find_resource_bar_recursive(child)
		if result:
			return result
	
	return null

func get_costs() -> Dictionary:
	"""Get building costs as a dictionary"""
	var costs = {}
	if gold_cost > 0:
		costs["gold"] = gold_cost
	if wood_cost > 0:
		costs["wood"] = wood_cost
	if stone_cost > 0:
		costs["stone"] = stone_cost
	if food_cost > 0:
		costs["food"] = food_cost
	return costs

func get_cost_display() -> String:
	"""Get formatted cost display string"""
	var costs = get_costs()
	var parts = []
	
	var icons = {
		"gold": "💰",
		"wood": "🪵",
		"stone": "🪨",
		"food": "🌾"
	}
	
	for resource_type in costs.keys():
		var icon = icons.get(resource_type, "❓")
		parts.append("%s %d" % [icon, costs[resource_type]])
	
	return " | ".join(parts)

func get_detailed_info() -> String:
	"""Get detailed building information"""
	var info = """[b]%s %s[/b]
%s

[b]Costs:[/b]
%s

[b]Build Time:[/b] %.1f seconds
[b]Health:[/b] %d
""" % [
		building_icon,
		building_name,
		building_description,
		_format_costs_detailed(),
		build_time,
		max_health
	]
	
	if provides_housing > 0:
		info += "\n[b]Provides Housing:[/b] %d" % provides_housing
	
	if not resource_generation.is_empty():
		info += "\n[b]Generates:[/b]"
		for resource_type in resource_generation.keys():
			var amount = resource_generation[resource_type]
			info += "\n  +%d %s/sec" % [amount, resource_type.capitalize()]
	
	return info

func _format_costs_detailed() -> String:
	"""Format costs with detailed breakdown"""
	var costs = get_costs()
	var parts = []
	
	var icons = {
		"gold": "💰",
		"wood": "🪵",
		"stone": "🪨",
		"food": "🌾"
	}
	
	for resource_type in costs.keys():
		var icon = icons.get(resource_type, "❓")
		parts.append("  %s %s: %d" % [icon, resource_type.capitalize(), costs[resource_type]])
	
	return "\n".join(parts)

func take_damage(damage: int):
	"""Apply damage to the building"""
	current_health = max(0, current_health - damage)
	
	if health_bar:
		health_bar.value = current_health
	
	if current_health <= 0:
		_destroy()

func repair(amount: int):
	"""Repair the building"""
	current_health = min(max_health, current_health + amount)
	
	if health_bar:
		health_bar.value = current_health

func _destroy():
	"""Destroy the building"""
	building_destroyed.emit()
	queue_free()

func _on_mouse_entered():
	"""Show cost label on hover"""
	if cost_label:
		cost_label.visible = true

func _on_mouse_exited():
	"""Hide cost label on hover end"""
	if cost_label:
		cost_label.visible = false

func _input_event(_viewport, event, _shape_idx):
	"""Handle input events (like clicking)"""
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		_select()

func _select():
	"""Select this building"""
	building_selected.emit()
	
	# Show detailed info
	var info_panel = _find_info_panel()
	if info_panel and info_panel.has_method("show_building_info"):
		info_panel.show_building_info(self)
	else:
		# Fallback: print to console
		print(get_detailed_info())

func _find_info_panel():
	"""Find the info panel in the scene tree"""
	var root = get_tree().root
	return root.get_node_or_null("InfoPanel")
