extends Building
# GoldMine.gd - Example building that generates gold

func _ready():
	building_name = "Gold Mine"
	building_icon = "⛏️"
	building_description = "Generates gold over time"
	
	gold_cost = 150
	wood_cost = 100
	stone_cost = 50
	
	build_time = 15.0
	max_health = 1500
	
	# Generate 10 gold per second
	resource_generation = {"gold": 10}
	
	super._ready()
