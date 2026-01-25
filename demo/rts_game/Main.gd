extends Node2D
# Main.gd - Main game scene that ties all UI components together

@onready var resource_bar: ResourceBar
@onready var worker_build_panel: WorkerBuildPanel
@onready var building_panel: BuildingPlacementPanel

func _ready():
	_setup_ui()
	_connect_signals()
	print("RTS Game Demo - Resource Pooling Clarity")
	print("=========================================")
	print("This demo addresses Linear issue ORC-153")
	print("")
	print("Features:")
	print("✓ Tutorial explaining resources")
	print("✓ Resource costs on buttons")
	print("✓ Highlight unaffordable items")
	print("✓ Income indicators (+X/sec)")
	print("✓ Resource usage descriptions")

func _setup_ui():
	"""Setup UI components"""
	# Create UI container
	var ui_layer = CanvasLayer.new()
	ui_layer.name = "UILayer"
	add_child(ui_layer)
	
	# Resource bar at the top
	resource_bar = ResourceBar.new()
	resource_bar.name = "ResourceBar"
	resource_bar.position = Vector2(20, 20)
	resource_bar.custom_minimum_size = Vector2(600, 80)
	ui_layer.add_child(resource_bar)
	
	# Building panel on the left
	building_panel = BuildingPlacementPanel.new()
	building_panel.name = "BuildingPanel"
	building_panel.position = Vector2(20, 120)
	building_panel.custom_minimum_size = Vector2(400, 400)
	building_panel.resource_bar = resource_bar
	ui_layer.add_child(building_panel)
	
	# Worker/unit panel on the right
	worker_build_panel = WorkerBuildPanel.new()
	worker_build_panel.name = "WorkerBuildPanel"
	worker_build_panel.position = Vector2(440, 120)
	worker_build_panel.custom_minimum_size = Vector2(350, 400)
	worker_build_panel.resource_bar = resource_bar
	ui_layer.add_child(worker_build_panel)
	
	# Instructions panel at the bottom
	_create_instructions_panel(ui_layer)

func _create_instructions_panel(parent: CanvasLayer):
	"""Create instructions panel"""
	var panel = PanelContainer.new()
	panel.position = Vector2(20, 540)
	panel.custom_minimum_size = Vector2(770, 100)
	parent.add_child(panel)
	
	var vbox = VBoxContainer.new()
	panel.add_child(vbox)
	
	var title = Label.new()
	title.text = "RTS Resource Management Demo - ORC-153"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 16)
	vbox.add_child(title)
	
	var instructions = Label.new()
	instructions.text = """Instructions:
• Hover over resources to see tooltips with descriptions and income rates
• Click unit/building buttons to train/build (costs shown on buttons)
• Unaffordable items are grayed out with shortage amounts displayed
• Watch resources grow automatically (+X/sec indicators)
• Build resource-generating buildings to increase income!"""
	instructions.add_theme_font_size_override("font_size", 12)
	vbox.add_child(instructions)

func _connect_signals():
	"""Connect signals from UI components"""
	if worker_build_panel:
		worker_build_panel.unit_training_started.connect(_on_unit_training_started)
		worker_build_panel.unit_training_complete.connect(_on_unit_training_complete)
		worker_build_panel.build_queue_updated.connect(_on_build_queue_updated)
	
	if building_panel:
		building_panel.building_placement_started.connect(_on_building_placement_started)
		building_panel.building_placed.connect(_on_building_placed)
		building_panel.placement_cancelled.connect(_on_placement_cancelled)
	
	if resource_bar:
		resource_bar.resource_changed.connect(_on_resource_changed)
		resource_bar.insufficient_resources.connect(_on_insufficient_resources)

# Signal handlers
func _on_unit_training_started(unit_type: String):
	print("[GAME] Training started: %s" % unit_type)

func _on_unit_training_complete(unit_type: String):
	print("[GAME] Training complete: %s" % unit_type)
	# In a real game, spawn the unit here

func _on_build_queue_updated(queue_size: int):
	print("[GAME] Build queue updated: %d items" % queue_size)

func _on_building_placement_started(building_type: String):
	print("[GAME] Placement mode: %s" % building_type)

func _on_building_placed(building_type: String, position: Vector2):
	print("[GAME] Building placed: %s at %s" % [building_type, position])
	# In a real game, instantiate the building here

func _on_placement_cancelled():
	print("[GAME] Placement cancelled")

func _on_resource_changed(resource_type: String, new_amount: int):
	# Optional: Add visual/audio feedback
	pass

func _on_insufficient_resources(resource_type: String, required: int, current: int):
	print("[GAME] Insufficient %s: have %d, need %d" % [resource_type, current, required])
