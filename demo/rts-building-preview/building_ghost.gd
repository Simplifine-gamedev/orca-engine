extends Node3D
class_name BuildingGhost

## Building Ghost/Preview Component
## Shows a semi-transparent preview of a building before placement
## 
## FIX FOR ORC-112:
## This script now correctly accepts and uses the faction parameter
## to show the appropriate faction-specific building model

var current_building_type: String = ""
var current_faction: FactionConfig.Faction = FactionConfig.Faction.HUMAN
var preview_mesh: MeshInstance3D
var is_valid_placement: bool = true

func _ready():
	# Create preview mesh
	preview_mesh = MeshInstance3D.new()
	add_child(preview_mesh)
	
	# Make it semi-transparent
	var material = StandardMaterial3D.new()
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.albedo_color = Color(1, 1, 1, 0.5)
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	preview_mesh.material_override = material

## Update the preview to show the correct building type for the current faction
## 
## THIS IS THE KEY FIX: Previously this function didn't accept or use the faction parameter
## causing it to always show human faction buildings
func update_preview(building_type: String, faction: FactionConfig.Faction):
	current_building_type = building_type
	current_faction = faction  # ← THE FIX: Now we store and use the faction!
	
	# Get the correct color for this faction and building type
	var color = FactionConfig.get_building_color(faction, building_type)
	
	# Update preview mesh
	_update_preview_mesh(building_type, color)
	
	print("BuildingGhost: Showing preview for ", FactionConfig.FACTION_NAMES[faction], 
		  " ", building_type)

## BEFORE (BUG): This was the old buggy version
## func update_preview(building_type: String):
##     # BUG: Always defaulted to human faction!
##     var color = FactionConfig.get_building_color(FactionConfig.Faction.HUMAN, building_type)
##     _update_preview_mesh(building_type, color)

func _update_preview_mesh(building_type: String, color: Color):
	# Create appropriate mesh based on building type
	var mesh: Mesh
	match building_type:
		FactionConfig.BARRACKS:
			# Square building
			mesh = BoxMesh.new()
			mesh.size = Vector3(3, 2, 3)
		FactionConfig.TOWN_HALL:
			# Large building
			mesh = BoxMesh.new()
			mesh.size = Vector3(5, 3, 5)
		FactionConfig.FARM:
			# Rectangular building
			mesh = BoxMesh.new()
			mesh.size = Vector3(4, 1.5, 3)
		_:
			mesh = BoxMesh.new()
			mesh.size = Vector3(2, 2, 2)
	
	preview_mesh.mesh = mesh
	
	# Apply faction-specific color with transparency
	var material = StandardMaterial3D.new()
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.albedo_color = color
	material.albedo_color.a = 0.5  # Semi-transparent
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	
	# Add a subtle emission to make it glow
	material.emission_enabled = true
	material.emission = color
	material.emission_energy_multiplier = 0.3
	
	preview_mesh.material_override = material

func set_valid_placement(valid: bool):
	is_valid_placement = valid
	
	# Change color based on validity
	if preview_mesh and preview_mesh.material_override:
		var material = preview_mesh.material_override as StandardMaterial3D
		if valid:
			# Keep faction color
			var color = FactionConfig.get_building_color(current_faction, current_building_type)
			material.albedo_color = color
			material.albedo_color.a = 0.5
		else:
			# Show red for invalid placement
			material.albedo_color = Color(1, 0, 0, 0.5)

func show_preview():
	visible = true

func hide_preview():
	visible = false

func _process(_delta):
	# Follow mouse position (simple implementation)
	# In a real game, this would raycast to terrain
	pass
