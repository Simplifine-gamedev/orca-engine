# Assets Needed for Archer & Crossbowman Units

This document lists all the 3D models, animations, textures, and other assets needed to complete the archer and crossbowman units implementation.

## 3D Models Required

### 1. Archer Character Model

**Description**: Medieval archer character with longbow
**Specifications**:
- Humanoid rig (compatible with Godot's skeleton system)
- Height: ~1.8m
- Polycount: 2000-5000 triangles (optimized for RTS camera distance)
- Materials: Leather armor, cloth tunic, wooden bow
- Color variants needed for 3 factions (Human, Orc, Elf)

**Components**:
- Character body with rigged skeleton
- Longbow weapon (can be separate object)
- Quiver with arrows (back attachment)

**Reference prompts for 3D generation**:
- "Low poly medieval archer character with longbow and quiver, wearing leather armor, suitable for RTS game"
- "Fantasy archer warrior in leather armor holding a wooden longbow, game-ready 3D model"

### 2. Crossbowman Character Model

**Description**: Medieval crossbowman character with heavy crossbow
**Specifications**:
- Humanoid rig (compatible with Godot's skeleton system)
- Height: ~1.8m
- Polycount: 2500-5500 triangles
- Materials: Padded armor, metal reinforcements, wooden crossbow
- Color variants needed for 3 factions

**Components**:
- Character body with rigged skeleton
- Heavy crossbow weapon (can be separate object)
- Bolt case (hip or back attachment)

**Reference prompts for 3D generation**:
- "Low poly medieval crossbowman character with heavy crossbow, wearing padded armor with metal plates, RTS game style"
- "Fantasy crossbowman warrior in medium armor holding a wooden crossbow, game-ready 3D model"

### 3. Archery Range Building Model

**Description**: Medieval military building for training ranged units
**Specifications**:
- Size: ~4m x 4m x 3m
- Polycount: 1000-3000 triangles
- Materials: Wood structure, thatch roof, training targets
- Faction variants: Human (standard), Orc (rugged), Elf (elegant)

**Features**:
- Visible training area
- Entrance/exit for units
- Targeting dummies or practice ranges
- Faction-appropriate architectural style

**Reference prompts for 3D generation**:
- "Medieval archery range building with wooden structure, thatch roof, and practice targets, RTS game building"
- "Fantasy military training building for archers, low poly game asset"

### 4. Projectile Models

#### Arrow
**Current**: Procedurally generated cylinder
**Needed**: Low-poly arrow model
- Length: ~0.8m
- Polycount: 50-100 triangles
- Components: Wooden shaft, metal tip, fletching
- Simple texture (wood grain, metal)

**Reference prompt**:
- "Simple low poly arrow projectile with wooden shaft and metal arrowhead for game"

#### Crossbow Bolt
**Current**: Procedurally generated cylinder
**Needed**: Low-poly crossbow bolt model
- Length: ~0.6m
- Polycount: 60-120 triangles
- Components: Thicker wooden shaft, larger metal tip, minimal fletching
- Simple texture (dark wood, iron tip)

**Reference prompt**:
- "Simple low poly crossbow bolt projectile with thick wooden shaft and iron tip for game"

## Animations Required

### Archer Animations

All animations should be compatible with humanoid rig:

1. **idle_bow** (loop)
   - Standing ready with bow held loosely
   - Slight breathing motion
   - Duration: 2-3 seconds

2. **walk_bow** (loop)
   - Walking forward with bow in hand
   - Smooth gait
   - Duration: 1 second per cycle

3. **attack_bow** (one-shot)
   - Draw arrow from quiver
   - Nock arrow
   - Draw bowstring
   - Release arrow
   - Return to ready position
   - Duration: 1.5 seconds

4. **death** (one-shot)
   - Fall backward or forward
   - Weapon drops
   - Duration: 1-2 seconds

### Crossbowman Animations

1. **idle_crossbow** (loop)
   - Standing ready with crossbow
   - Duration: 2-3 seconds

2. **walk_crossbow** (loop)
   - Walking with crossbow
   - Duration: 1 second per cycle

3. **attack_crossbow** (one-shot)
   - Aim crossbow
   - Pull trigger
   - Recoil animation
   - Duration: 1 second

4. **reload_crossbow** (one-shot)
   - Lower crossbow
   - Pull back string mechanism
   - Load bolt
   - Raise to ready
   - Duration: 1.2 seconds

5. **death** (one-shot)
   - Fall animation
   - Duration: 1-2 seconds

## UI Icons Needed

### Unit Icons (64x64 or 128x128)

1. **Archer Icon**
   - Portrait or silhouette of archer with bow
   - Variants: Human, Orc, Elf

2. **Crossbowman Icon**
   - Portrait or silhouette of crossbowman with crossbow
   - Variants: Human, Orc, Elf

### Building Icons

3. **Archery Range Icon**
   - Building icon or symbol
   - Variants: Human (Archery Range), Orc (War Lodge), Elf (Hunter's Hall)

### Projectile Icons (optional, for UI/abilities)

4. **Arrow Icon** - Small arrow symbol
5. **Crossbow Bolt Icon** - Crossbow bolt symbol

## Textures & Materials

### Character Textures
- Diffuse/Albedo maps (1024x1024 or 512x512)
- Normal maps (optional, for detail)
- Faction color masks for easy recoloring

### Building Textures
- Diffuse/Albedo maps (512x512 or 1024x1024)
- Faction variants

### Weapon Textures
- Wood grain for bows/crossbows
- Metal for arrow/bolt tips

## Sound Effects Needed

### Unit Sounds

1. **Archer**
   - Bow draw sound (creak of wood and string)
   - Arrow release (twang)
   - Arrow impact (thud for flesh, clang for armor)
   - Footsteps (leather boots on various surfaces)
   - Selection sounds (voice acknowledgments)
   - Death sound

2. **Crossbowman**
   - Crossbow aim/ready sound
   - Crossbow fire (mechanical click + release)
   - Reload sound (ratcheting mechanism)
   - Bolt impact (heavy thud)
   - Footsteps
   - Selection sounds
   - Death sound

### Building Sounds

3. **Archery Range**
   - Training ambient sounds (arrows hitting targets)
   - Unit spawned sound
   - Building damaged/destroyed sounds

## Particle Effects Needed

### Projectile Effects

1. **Arrow Trail**
   - Subtle motion blur or trail
   - Very light particle stream

2. **Crossbow Bolt Trail**
   - Slightly more pronounced than arrow

3. **Impact Effects**
   - Small dust/blood splash on hit
   - Sparks if hitting metal
   - Wood splinters if hitting shields

### Unit Effects

4. **Selection Indicator**
   - Ring or circle under selected units
   - Faction color

5. **Health Bar** (optional)
   - Above unit when damaged

## Generating Assets with Orca Engine

The Orca Engine includes AI-powered asset generation. To generate 3D models:

### Using the Editor

1. Open the AI chat dock in Orca Engine editor
2. Use prompts like:
   ```
   Generate a 3D model of a medieval archer character with a longbow, 
   wearing leather armor, optimized for RTS game use
   ```

### Using the Backend API

If 3D generation is configured in the backend:

```bash
curl -X POST http://localhost:8000/api/3d/generate/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "medieval archer character with longbow"}'
```

See `backend/3d-generation.md` for configuration details.

## Asset Import Guidelines

### 3D Models
- Format: `.glb` or `.gltf` preferred for Godot
- Scale: 1 unit = 1 meter in Godot
- Origin: At character's feet (ground level)
- Rig: Godot-compatible humanoid skeleton

### Textures
- Format: `.png` for color/alpha, `.jpg` for diffuse
- Power of 2 dimensions (512x512, 1024x1024, etc.)
- Include normal maps if available

### Animations
- Baked into model file or separate `.res` files
- Frame rate: 30 FPS
- Use Godot's AnimationPlayer system

### Icons
- Format: `.png` with transparency
- Size: 128x128 recommended
- Clear silhouettes work best

## Priority Order

**Phase 1 - Basic Functionality** (Current - Placeholder models work)
- ✅ Unit logic and combat
- ✅ Projectile physics
- ✅ Building training system
- ✅ Configuration files

**Phase 2 - Core Visuals** (Next priority)
1. Archer character model
2. Crossbowman character model
3. Basic animations (idle, walk, attack)
4. Arrow and bolt models

**Phase 3 - Polish**
1. Archery Range building model
2. Additional animations (death, reload)
3. UI icons
4. Sound effects

**Phase 4 - Enhancement**
1. Particle effects
2. Faction variants
3. Advanced animations
4. Polished textures

## Notes

- Current implementation uses procedural placeholder models (capsules) that work functionally
- All scripts are complete and functional
- Adding proper models and animations is primarily an art asset task
- Models can be generated using Orca Engine's AI tools or traditional 3D modeling software
- Faction variations can share base models with material/color swaps
