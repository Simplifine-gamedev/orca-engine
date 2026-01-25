# RTS Loot Distribution - Quick Start

## TL;DR for RTS Game Team

### Answer to Haridzieko's Question
**"Is it intentional that all the good loot is in the middle?"**

**YES** ✅ - It's intentional and creates great gameplay (encourages center exploration, ensures player fights).

---

## Files You Need

1. **`rts_loot_distribution_example.gd`** - Copy this into your game
2. **`RTS_LOOT_DISTRIBUTION_GUIDE.md`** - Full documentation
3. **`rts_loot_distribution_test.gd`** - Test/visualize patterns

---

## 30-Second Integration

```gdscript
# In your map initialization script:
var loot_system = RTSLootDistribution.new()
add_child(loot_system)

loot_system.map_size = Vector2(2000, 2000)  # Your map size
loot_system.distribution_pattern = RTSLootDistribution.DistributionPattern.CENTER_FOCUSED
loot_system.generate_loot_distribution()

# Spawn loot items
for loot_data in loot_system.get_loot_positions():
    spawn_loot(loot_data.position, loot_data.tier)
```

---

## Recommended Settings

```gdscript
distribution_pattern = CENTER_FOCUSED  # Keep this
center_weight = 2.5                    # Adjust 2.0-3.0 if needed
map_size = Vector2(2000, 2000)         # Match your map
total_loot_items = 150                 # Scale with map size
```

---

## Distribution Patterns Available

| Pattern | Use Case |
|---------|----------|
| **CENTER_FOCUSED** | **Default** - Current design (recommended) |
| UNIFORM | Even spread, less combat |
| EDGE_FOCUSED | Reverse - edges have more |
| QUADRANT | Team territories |
| RING_PATTERN | Staged progression |

---

## Testing

1. Open Godot/Orca Engine
2. Create a 2D scene
3. Attach `rts_loot_distribution_test.gd`
4. Run scene
5. Press keys 1-5 to switch patterns
6. Press +/- to adjust center weight

---

## Support

See `RTS_LOOT_DISTRIBUTION_GUIDE.md` for:
- Detailed explanation
- Advanced tuning
- Scaling formulas
- Best practices

---

## Branch Info

**Branch**: `cursor/ORC-135-map-loot-distribution-116f`
**Commits**:
- `c2efc508` - Core implementation
- `71357cf3` - Documentation summary

Ready to merge/integrate! 🚀
