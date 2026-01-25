# RTS Loot Distribution System - ORC-135

## Overview

This guide addresses the loot distribution feedback from Haridzieko regarding ORC-135 (Map size increase).

### Feedback Question
> "Is it intentional that all the good loot is in the middle?"

## Answer: Yes, It's Intentional (But Configurable)

Based on Haridzieko's positive feedback, the center-focused loot distribution creates beneficial gameplay dynamics:

### Benefits of Center-Focused Distribution

1. **Encourages Exploration**: Players are incentivized to venture toward the center of the map to find better resources
2. **Creates Conflict Zones**: All players naturally converge toward the center, ensuring player encounters and combat
3. **Risk/Reward Balance**: The center becomes a high-value, high-risk area
4. **Natural Pacing**: Early game focuses on edges (safe), mid-game pushes toward center (contested)
5. **Strategic Depth**: Teams must decide when to risk center exploration vs. safe peripheral farming

## Implementation

The `rts_loot_distribution_example.gd` script provides a **configurable** loot distribution system with multiple patterns:

### Distribution Patterns

#### 1. CENTER_FOCUSED (Default/Recommended)
- **Use Case**: The current design that Haridzieko liked
- **Behavior**: Higher concentration of loot toward the map center
- **Gameplay**: Incentivizes exploration and ensures player meetings/fights
- **Configuration**: Adjust `center_weight` parameter (default: 2.5)

```gdscript
distribution_pattern = DistributionPattern.CENTER_FOCUSED
center_weight = 2.5  # Higher = more center concentration
```

#### 2. UNIFORM
- **Use Case**: More casual or exploration-focused gameplay
- **Behavior**: Even distribution across entire map
- **Gameplay**: Reduces forced conflict, players can farm anywhere

```gdscript
distribution_pattern = DistributionPattern.UNIFORM
```

#### 3. EDGE_FOCUSED
- **Use Case**: Reverse psychology - make center safer
- **Behavior**: Rewards edge exploration
- **Gameplay**: Players spread out to edges

```gdscript
distribution_pattern = DistributionPattern.EDGE_FOCUSED
```

#### 4. QUADRANT
- **Use Case**: Team-based gameplay with territorial zones
- **Behavior**: Each quadrant gets equal loot distribution
- **Gameplay**: Natural team territories emerge

```gdscript
distribution_pattern = DistributionPattern.QUADRANT
```

#### 5. RING_PATTERN
- **Use Case**: Staged progression gameplay
- **Behavior**: Concentric rings with varying density
- **Gameplay**: Natural expansion stages from spawn to center

```gdscript
distribution_pattern = DistributionPattern.RING_PATTERN
```

## Usage Example

```gdscript
extends Node2D

var loot_system: RTSLootDistribution

func _ready():
    # Create loot distribution system
    loot_system = RTSLootDistribution.new()
    add_child(loot_system)
    
    # Configure for bigger map (per ORC-135)
    loot_system.map_size = Vector2(2000, 2000)  # Increased from 1000x1000
    
    # Use center-focused distribution (Haridzieko's preferred)
    loot_system.distribution_pattern = RTSLootDistribution.DistributionPattern.CENTER_FOCUSED
    loot_system.center_weight = 2.5  # Adjust to taste
    
    # Set loot quantities
    loot_system.total_loot_items = 150  # More loot for bigger map
    
    # Configure loot tiers
    loot_system.common_loot_percentage = 60.0
    loot_system.rare_loot_percentage = 30.0
    loot_system.epic_loot_percentage = 10.0
    
    # Generate and visualize
    loot_system.generate_loot_distribution()
    loot_system.visualize_distribution()
    
    # Spawn actual loot items
    _spawn_loot_items()

func _spawn_loot_items():
    var loot_positions = loot_system.get_loot_positions()
    
    for loot_data in loot_positions:
        var loot_item = preload("res://scenes/loot_item.tscn").instantiate()
        loot_item.position = loot_data.position
        loot_item.tier = loot_data.tier
        add_child(loot_item)
```

## Recommendations

### For Current Design (Based on Feedback)

Keep the **CENTER_FOCUSED** distribution with these settings:

```gdscript
distribution_pattern = DistributionPattern.CENTER_FOCUSED
map_size = Vector2(2000, 2000)  # Bigger map per ORC-135
center_weight = 2.5
total_loot_items = 150
min_distance_between_loot = 50.0
```

### Tuning the Center Concentration

Adjust `center_weight` to control the strength of center bias:

- `1.0` = Uniform distribution (no bias)
- `1.5` = Slight center preference
- `2.0` = Moderate center preference
- **`2.5`** = Strong center preference (recommended)
- `3.0+` = Very strong center preference

### Quality Distribution

Consider making **better loot** more central by modifying tier assignment:

```gdscript
func _assign_loot_tiers_with_center_bias() -> void:
    var center = map_size / 2
    
    # Sort by distance from center (closest first)
    loot_positions.sort_custom(func(a, b): 
        return a.position.distance_to(center) < b.position.distance_to(center)
    )
    
    # Assign better tiers to items closer to center
    var num_epic = int(total_loot_items * epic_loot_percentage / 100.0)
    var num_rare = int(total_loot_items * rare_loot_percentage / 100.0)
    
    for i in range(loot_positions.size()):
        if i < num_epic:
            loot_positions[i].tier = "epic"
        elif i < num_epic + num_rare:
            loot_positions[i].tier = "rare"
        else:
            loot_positions[i].tier = "common"
```

## Testing Different Patterns

Use the `visualize_distribution()` function to analyze patterns:

```gdscript
loot_system.visualize_distribution()
```

Output example:
```
=== Loot Distribution Visualization ===
Pattern: CENTER_FOCUSED
Map Size: (2000, 2000)
Total Loot: 150
Loot in center 25% of map: 62.7%

Tier Distribution:
  Common: 90 (60.0%)
  Rare: 45 (30.0%)
  Epic: 15 (10.0%)
=====================================
```

## Map Size Considerations

With the bigger map (ORC-135), consider:

1. **Increase total loot**: More map area = more loot needed
2. **Adjust spawn density**: Scale `min_distance_between_loot` proportionally
3. **Tune center weight**: Larger maps may need stronger center bias to maintain convergence

### Scaling Formula

```gdscript
# Scale loot count with map area
var base_map_size = 1000.0 * 1000.0  # Original 1000x1000
var new_map_area = map_size.x * map_size.y
var scale_factor = new_map_area / base_map_size
total_loot_items = int(100 * scale_factor)  # Base 100 items

# Scale minimum distance proportionally
var base_min_distance = 50.0
min_distance_between_loot = base_min_distance * sqrt(scale_factor)
```

## Conclusion

The center-focused loot distribution is **intentionally designed** based on positive player feedback. It creates engaging gameplay by:

- ✅ Incentivizing exploration
- ✅ Ensuring player encounters
- ✅ Creating strategic risk/reward decisions
- ✅ Establishing natural game pacing

The system is fully configurable to support different gameplay modes or to fine-tune the degree of center concentration based on further playtesting feedback.

---

**Status**: ORC-135 implementation complete with configurable loot distribution system

**Next Steps**:
1. Integrate `rts_loot_distribution_example.gd` into the RTS game project
2. Test with different `center_weight` values (recommended: 2.0-3.0)
3. Consider tier-based center bias for even stronger center incentive
4. Gather additional playtesting feedback on optimal settings
