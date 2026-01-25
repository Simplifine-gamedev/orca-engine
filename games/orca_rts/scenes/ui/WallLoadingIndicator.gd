extends Control
class_name WallLoadingIndicator

## Loading indicator UI for wall system assets
## Shows visual feedback while wall preview assets are loading

@onready var progress_bar: ProgressBar = $ProgressBar
@onready var loading_label: Label = $LoadingLabel
@onready var loading_spinner: TextureRect = $LoadingSpinner

var wall_system: WallSystem = null
var is_showing: bool = false

func _ready():
	# Hide by default
	visible = false
	
	# Create UI elements if they don't exist (for runtime creation)
	if not has_node("ProgressBar"):
		_create_ui_elements()
	
	# Find wall system in the scene
	_find_wall_system()

func _create_ui_elements():
	# Create progress bar
	progress_bar = ProgressBar.new()
	progress_bar.name = "ProgressBar"
	progress_bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	progress_bar.custom_minimum_size = Vector2(300, 30)
	progress_bar.show_percentage = true
	add_child(progress_bar)
	
	# Create loading label
	loading_label = Label.new()
	loading_label.name = "LoadingLabel"
	loading_label.text = "Loading wall assets..."
	loading_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	add_child(loading_label)
	
	# Create loading spinner (placeholder)
	loading_spinner = TextureRect.new()
	loading_spinner.name = "LoadingSpinner"
	loading_spinner.custom_minimum_size = Vector2(32, 32)
	add_child(loading_spinner)
	
	# Layout
	var vbox = VBoxContainer.new()
	vbox.name = "VBox"
	add_child(vbox)
	move_child(vbox, 0)
	
	vbox.add_child(loading_spinner)
	vbox.add_child(loading_label)
	vbox.add_child(progress_bar)
	
	vbox.size_flags_horizontal = Control.SIZE_SHRINK_CENTER
	vbox.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	vbox.position = Vector2(
		(get_viewport().size.x - vbox.size.x) / 2,
		(get_viewport().size.y - vbox.size.y) / 2
	)

func _find_wall_system():
	# Try to find WallSystem node in the scene tree
	wall_system = get_node_or_null("/root/WallSystem")
	
	if not wall_system:
		# Try to find it as a child of any node
		wall_system = _find_node_by_class(get_tree().root, "WallSystem")
	
	if wall_system:
		# Connect to wall system signals
		wall_system.wall_preview_loading_started.connect(_on_loading_started)
		wall_system.wall_preview_loaded.connect(_on_loading_finished)
		print("[LoadingIndicator] Connected to WallSystem")
	else:
		push_warning("[LoadingIndicator] Could not find WallSystem in scene tree")

func _find_node_by_class(node: Node, class_name: String) -> Node:
	if node.get_class() == class_name or (node.get_script() and node.get_script().get_global_name() == class_name):
		return node
	
	for child in node.get_children():
		var result = _find_node_by_class(child, class_name)
		if result:
			return result
	
	return null

func _process(_delta):
	if not is_showing or not wall_system:
		return
	
	# Update progress bar
	var progress = wall_system.get_loading_progress()
	if progress_bar:
		progress_bar.value = progress * 100.0
	
	# Rotate spinner for visual feedback
	if loading_spinner:
		loading_spinner.rotation += _delta * 3.0

func _on_loading_started():
	show_loading()

func _on_loading_finished():
	hide_loading()

## Show the loading indicator
func show_loading():
	visible = true
	is_showing = true
	if loading_label:
		loading_label.text = "Loading wall assets..."
	if progress_bar:
		progress_bar.value = 0.0
	print("[LoadingIndicator] Showing loading indicator")

## Hide the loading indicator
func hide_loading():
	visible = false
	is_showing = false
	print("[LoadingIndicator] Hiding loading indicator")

## Manually set wall system reference
func set_wall_system(system: WallSystem):
	wall_system = system
	if wall_system:
		wall_system.wall_preview_loading_started.connect(_on_loading_started)
		wall_system.wall_preview_loaded.connect(_on_loading_finished)
