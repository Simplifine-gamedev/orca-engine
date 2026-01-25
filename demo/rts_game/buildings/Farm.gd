extends Building
# Farm.gd - Generates food resources

func _ready():
	building_name = "Farm"
	building_icon = "🌾"
	building_description = "Produces food for your units"
	
	gold_cost = 80
	wood_cost = 60
	
	build_time = 12.0
	max_health = 800
	
	# Generate 5 food per second
	resource_generation = {"food": 5}
	
	super._ready()
