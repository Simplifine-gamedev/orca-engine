extends WorldEnvironment
## Improved environment setup for better RTS game visibility
## Addresses ORC-142: Game looks dark, units hard to see

@export_group("Ambient Lighting")
@export var ambient_light_color := Color(0.8, 0.85, 0.9)  # Slightly blue-tinted for outdoor feel
@export_range(0.0, 3.0, 0.1) var ambient_light_energy := 1.5  # Increased from typical 1.0
@export_range(0.0, 1.0, 0.05) var sky_contribution := 0.3

@export_group("Directional Light")
@export var sun_color := Color(1.0, 0.95, 0.85)  # Warm sunlight
@export_range(0.0, 5.0, 0.1) var sun_energy := 1.8  # Brighter sun
@export_range(-90, 90, 1) var sun_angle_degrees := -45.0  # Angle from horizontal

@export_group("Fog Settings")
@export var enable_fog := true
@export var fog_color := Color(0.7, 0.75, 0.8)  # Light blue-grey
@export_range(0.0, 1000.0, 10.0) var fog_start_distance := 150.0
@export_range(0.0, 2000.0, 10.0) var fog_end_distance := 800.0

@export_group("Shadow Quality")
@export var shadow_quality := true  # Use higher quality shadows
@export_range(0.0, 1.0, 0.05) var shadow_opacity := 0.6  # Lighter shadows for better visibility

@onready var directional_light: DirectionalLight3D = $"../DirectionalLight3D"

func _ready():
	setup_environment()
	setup_directional_light()

func setup_environment():
	"""Configure environment for optimal visibility"""
	if not environment:
		environment = Environment.new()
	
	# Ambient light - crucial for unit visibility
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = ambient_light_color
	environment.ambient_light_energy = ambient_light_energy
	environment.ambient_light_sky_contribution = sky_contribution
	
	# Background
	environment.background_mode = Environment.BG_SKY
	environment.background_energy_multiplier = 1.2
	
	# Fog for depth perception without obscuring units
	if enable_fog:
		environment.fog_enabled = true
		environment.fog_light_color = fog_color
		environment.fog_density = 0.001
		environment.fog_aerial_perspective = 0.5
	
	# Tone mapping for better color visibility
	environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	environment.tonemap_exposure = 1.1
	environment.tonemap_white = 1.0
	
	# Slight glow to make units pop
	environment.glow_enabled = true
	environment.glow_intensity = 0.3
	environment.glow_strength = 0.8
	environment.glow_bloom = 0.1
	environment.glow_blend_mode = Environment.GLOW_BLEND_MODE_SOFTLIGHT
	
	# SSAO for depth without making scene too dark
	environment.ssao_enabled = true
	environment.ssao_radius = 2.0
	environment.ssao_intensity = 1.0
	environment.ssao_detail = 0.5
	environment.ssao_light_affect = 0.3  # Reduced to prevent darkening
	
	# Sky
	if not environment.sky:
		var sky = Sky.new()
		var sky_material = ProceduralSkyMaterial.new()
		sky_material.sky_top_color = Color(0.4, 0.6, 0.9)
		sky_material.sky_horizon_color = Color(0.6, 0.7, 0.8)
		sky_material.ground_bottom_color = Color(0.2, 0.25, 0.3)
		sky_material.ground_horizon_color = Color(0.5, 0.55, 0.6)
		sky_material.sun_angle_max = 30.0
		sky.sky_material = sky_material
		environment.sky = sky

func setup_directional_light():
	"""Configure main directional light (sun)"""
	if not directional_light:
		return
	
	directional_light.light_color = sun_color
	directional_light.light_energy = sun_energy
	directional_light.rotation_degrees.x = sun_angle_degrees
	directional_light.rotation_degrees.y = -45.0  # Southwest direction
	
	# Shadow configuration for better visibility
	directional_light.shadow_enabled = true
	if shadow_quality:
		directional_light.directional_shadow_mode = DirectionalLight3D.SHADOW_PARALLEL_4_SPLITS
		directional_light.directional_shadow_max_distance = 200.0
	else:
		directional_light.directional_shadow_mode = DirectionalLight3D.SHADOW_PARALLEL_2_SPLITS
	
	directional_light.shadow_opacity = shadow_opacity
	directional_light.shadow_blur = 1.5  # Softer shadows
	
	# Reduce shadow acne
	directional_light.shadow_normal_bias = 2.0
	directional_light.shadow_bias = 0.1

func set_time_of_day(hour: float):
	"""Adjust lighting based on time of day (0-24)"""
	# Adjust sun angle
	var angle = -90.0 + (hour / 24.0) * 180.0
	sun_angle_degrees = clamp(angle, -90, 90)
	
	# Adjust colors and intensity
	if hour >= 6 and hour <= 18:  # Daytime
		sun_energy = lerp(1.0, 2.0, (hour - 6) / 6.0) if hour < 12 else lerp(2.0, 1.0, (hour - 12) / 6.0)
		ambient_light_energy = 1.5
	else:  # Night
		sun_energy = 0.1
		ambient_light_energy = 0.5
		ambient_light_color = Color(0.3, 0.35, 0.5)  # Cooler at night
	
	setup_directional_light()
	setup_environment()
