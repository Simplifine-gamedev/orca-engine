# Advanced Indexing System

Orca Engine features a best-in-class Godot indexing system that understands code at the **function level** and tracks **signal flows** and **dependencies** across your entire project.

## Architecture Overview

The indexing system has evolved from basic line-based chunking to **semantic understanding** of Godot projects:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Change   │ -> │  Smart Chunking   │ -> │  Weaviate DB    │
│   Detection     │    │  + Dependencies   │    │  + Graph Data   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                       │
        v                        v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Real-time      │    │  Function-Level  │    │  Multi-Hop      │
│  Updates        │    │  Intelligence    │    │  Tracing        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Enhanced Features

### 🎯 **Function-Level Chunking**

**Before**: Files split into 50-line chunks
**Now**: Each semantic unit (function, signal, export) is a searchable chunk

#### GDScript Intelligence
```gdscript
# Each of these becomes a separate searchable chunk:
func jump():              # ← Function chunk with metadata
    emit_signal("jumped") # ← Signal emission tracked
    
signal health_changed     # ← Signal definition chunk

@export var speed: float  # ← Export variable chunk
```

#### Metadata Extraction
Each function chunk includes:
- **Signals emitted**: `emit_signal()` calls
- **Functions called**: Direct and `self.method()` calls  
- **Nodes accessed**: `get_node()`, `$Node` references
- **Physics APIs**: `move_and_slide()`, `is_on_floor()`, etc.

#### Scene File Intelligence
`.tscn` files are parsed into **node-level chunks**:
```ini
[node name="Player" type="CharacterBody2D"]  # ← Player node chunk
script = ExtResource("1_abc123")             # ← Script attachment tracked

[node name="CollisionShape2D" parent="Player"] # ← Child node chunk

[connection signal="body_entered" from="Area2D" to="Player" method="take_damage"]
# ↑ Signal connection tracked with full flow information
```

### 🔗 **Signal Flow Tracking**

The system understands signal chains across your entire project:

```gdscript
# In Enemy.gd
func _on_death():
    emit_signal("enemy_died", position)  # ← Emission tracked

# In GameManager.gd  
func _ready():
    enemy.connect("enemy_died", _on_enemy_died)  # ← Connection tracked

func _on_enemy_died(pos):  # ← Handler method tracked
    spawn_explosion(pos)
```

**Result**: Search for "enemy death" finds the complete signal flow chain.

### 🕸️ **Multi-Hop Dependency Tracing**

Trace function call chains and impact analysis:

```gdscript
# Query: "What affects player movement?"
# Traces: Input → PlayerController → CharacterBody2D → Physics → Collisions

func _physics_process():     # ← Starting point
    handle_input()          # ← 1-hop dependency
    apply_gravity()         # ← 1-hop dependency  
    move_and_slide()        # ← Physics API call
    update_animation()      # ← 1-hop dependency

func handle_input():
    if Input.is_action_pressed("jump"):  # ← 2-hop: input system
        try_jump()                       # ← 2-hop: function call
```

**Dependency Types Tracked**:
- `CALLS_FUNCTION`: Direct function calls
- `EMITS_SIGNAL`: Signal emissions
- `CONNECTS_SIGNAL`: Signal connections  
- `ACCESSES_NODE`: Node path references
- `USES_PHYSICS_API`: Physics method calls
- `ACCESSES_INPUT`: Input system calls

### 🔍 **Enhanced Search Modes**

#### 1. Semantic Search (Default)
Understanding-based search using AI embeddings:
```
Query: "player collision physics"
Finds: Collision detection code, physics bodies, damage systems
```

#### 2. Keyword Search  
Exact text matching for precise lookups:
```  
Query: "add_vectors"
Finds: Only files containing exactly "add_vectors"
```

#### 3. Hybrid Search
Combines both approaches for comprehensive results:
```
Query: "jump function implementation"  
Finds: Semantic matches + exact "jump" occurrences
```

### 📊 **Graph Intelligence**

#### Centrality Analysis
Files are ranked by **structural importance**:
- **High centrality**: `project.godot`, main scenes, autoloads
- **Medium centrality**: Shared utilities, base classes
- **Low centrality**: Specific feature implementations

#### Architectural Role Detection
```
- Hub: Files with many connections (GameManager.gd)
- Utility: Helper functions and tools  
- Feature: Specific game mechanics
- Config: Settings and project files
- UI: Interface and menu systems
```

#### Cross-Section Deduplication
Results are intelligently deduplicated:
- **Similar Files**: Semantic matches
- **Central Files**: Architecturally important (no duplicates)
- **Dependencies**: Multi-hop tracing results

## Weaviate Integration

### Collections Structure

#### ProjectEmbedding
Stores function-level chunks with embeddings:
```json
{
  "file_path": "player/player.gd",
  "content": "func jump():\n    velocity.y = JUMP_VELOCITY\n    emit_signal('jumped')",
  "chunk_type": "function",
  "function_name": "jump",
  "signals_emitted": ["jumped"],
  "functions_called": [],
  "nodes_accessed": [],
  "chunk_start": 45,
  "chunk_end": 48
}
```

#### ProjectGraph  
Traditional file-to-file relationships:
```json
{
  "source_file": "level.tscn", 
  "target_file": "player.tscn",
  "relationship_type": "INSTANTIATES_SCENE",
  "weight": 2.0
}
```

#### ProjectDependencies
Detailed function-level dependencies:
```json
{
  "source_file": "player.gd",
  "source_function": "_physics_process", 
  "target_function": "move_and_slide",
  "dependency_type": "CALLS_FUNCTION",
  "line_number": 67,
  "context": "move_and_slide()",
  "weight": 1.0
}
```

## Real-Time Updates

### File Change Detection
- **Godot Plugin**: Detects file saves and sends to backend
- **Smart Filtering**: Ignores `.import` files and binary assets
- **Incremental Updates**: Only re-processes changed files

### Update Pipeline
1. **File Change** → **Remove old chunks** → **Re-extract dependencies**
2. **Smart Chunking** → **Generate embeddings** → **Store in Weaviate** 
3. **Graph Updates** → **Dependency Tracing** → **Ready for Search**

### Performance
- **Function chunking**: ~2.7x more intelligent chunks from same files
- **Parallel processing**: Background dependency extraction
- **Smart caching**: Avoids redundant re-processing

## Search Intelligence Examples

### Complex Game Mechanics
```
Query: "how does player take damage from enemies"
Finds:
├── Enemy collision detection (Area2D, body_entered signal)
├── Player health system (health variable, take_damage function) 
├── UI health bar updates (signal connections to UI)
└── Game over logic (health <= 0 conditions)
```

### System Integration
```
Query: "what happens when player jumps"  
Traces:
Input System → Player Controller → Physics Body → Animation Tree → Audio System
```

### Architecture Analysis  
```
Query: "find all singletons and autoloads"
Returns:
├── project.godot autoload declarations
├── Singleton script implementations  
└── Usage patterns across the project
```

## API Integration

### Search with Dependency Tracing
```http
POST /search_across_project
{
  "query": "player movement system", 
  "trace_dependencies": true,
  "search_mode": "hybrid",
  "max_results": 10
}
```

### Response Structure
```json
{
  "success": true,
  "search_mode": "hybrid", 
  "results": [
    {
      "file_path": "player.gd",
      "content": "func _physics_process()...",
      "chunk_type": "function", 
      "function_name": "_physics_process",
      "dependencies": ["handle_input", "apply_gravity", "move_and_slide"],
      "signals_emitted": ["moved", "landed"],
      "similarity": 0.94
    }
  ],
  "graph": {
    "player.gd": {
      "nodes": [...],
      "edges": [...], 
      "signal_flows": [...]
    }
  }
}
```

## Configuration

### Environment Variables
```bash
WEAVIATE_URL=https://your-cluster.weaviate.cloud
WEAVIATE_API_KEY=your-api-key
OPENAI_API_KEY=your-openai-key  # For embeddings
```

### Index Configuration
```python
# Supported file types for indexing
INDEXED_EXTENSIONS = {
    '.gd', '.cs', '.cpp', '.h', '.hpp', '.c',
    '.tscn', '.tres', '.res', '.godot', 
    '.json', '.cfg', '.md', '.txt',
    '.shader', '.gdshader', '.glsl'
}

# Filtered out (never indexed)
SKIP_EXTENSIONS = {
    '.png', '.jpg', '.mp3', '.wav', '.import', '.uid'
}
```

## Performance Metrics

### Typical Project (50 files)
- **Basic system**: 150 line-based chunks
- **Enhanced system**: 400+ function-level chunks
- **Index time**: ~10-15 seconds  
- **Search response**: <200ms
- **Dependencies tracked**: 50-200 per project

### Large Project (500 files)
- **Function chunks**: 2000-5000
- **Dependencies**: 500-2000
- **Index time**: 2-5 minutes
- **Search performance**: Maintained <500ms

## Troubleshooting

### Common Issues

**No search results**:
```bash
# Check if files are being filtered
grep "Skipping filtered file" logs

# Verify Weaviate connection  
grep "Weaviate v4 client connected" logs
```

**Poor dependency tracking**:
```bash  
# Check function detection
grep "Found [0-9]+ functions" logs

# Verify dependency extraction
grep "[0-9]+ dependencies" logs
```

**Slow indexing**:
- Large files with many functions take longer
- Network latency to Weaviate cluster
- Consider increasing chunk size limits

### Debug Logging
Enable verbose logging:
```python
# In weaviate_vector_manager.py
DEBUG_CHUNKING = True    # Shows chunk details
DEBUG_DEPENDENCIES = True  # Shows dependency extraction  
DEBUG_SEARCH = True      # Shows search internals
```

## Future Enhancements

### Planned Features
- **Cross-language support**: C# and C++ function parsing
- **Visual scripting**: VisualScript node understanding  
- **Asset relationships**: Texture/mesh/audio dependencies
- **Performance profiling**: Identify bottleneck functions
- **Refactoring assistance**: Safe rename across dependencies

### Integration Opportunities  
- **LSP integration**: Real-time code intelligence
- **Git integration**: Track changes in function dependencies
- **Testing**: Identify test coverage gaps by function
- **Documentation**: Auto-generate docs from function relationships
