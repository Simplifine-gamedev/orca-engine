extends Building
# Barracks.gd - Building for training military units

func _ready():
	building_name = "Barracks"
	building_icon = "🏰"
	building_description = "Train soldiers and military units"
	
	gold_cost = 200
	wood_cost = 150
	stone_cost = 100
	
	build_time = 20.0
	max_health = 2000
	provides_housing = 10
	
	super._ready()
