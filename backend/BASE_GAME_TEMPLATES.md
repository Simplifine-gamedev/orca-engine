# Base Game Template Selector

This feature enables the AI agent to browse and install curated open-source Godot 4 game frameworks from a catalog of high-quality starter templates.

## Overview

The base game template selector gives your AI agent the ability to:
- **List** available game templates filtered by category
- **Install** templates by downloading and extracting them to a target directory
- Work with vetted, open-source templates with MIT/CC0/Unlicense licensing

## Available Templates

### FPS (First-Person Shooter)
- **kenney-fps**: Kenney's Starter Kit FPS with character controller, weapon switching, enemies, CC0 assets

### 2D Platformer
- **g2p-2d-platformer**: Complete 2D platformer with levels, menu, scoring, SFX
- **adildev-2d-platformer**: Clean modular 2D platformer template

### 3D Platformer / Third-Person
- **kenney-3d-platformer**: Kenney's 3D Platformer with double-jump, collectibles, falling platforms
- **gdquest-third-person**: Professional third-person controller with combat mechanics

### City Building
- **kenney-city-builder**: City builder with build/remove structures, save/load, dynamic MeshLibrary

### Slasher / Hack-and-Slash
- **cats-soulslike**: Modular Souls-like template with combat, stamina, dodge/roll mechanics

## Usage in AI Agent

The AI agent can now use the `project_manager` tool with these new operations:

### List Templates

```json
{
  "op": "templates.list"
}
```

**Optional Parameters:**
- `template_category`: Filter by category (`"fps"`, `"2d-platformer"`, `"3d-platformer"`, `"city-building"`, `"slasher"`)

**Response:**
```json
{
  "success": true,
  "templates": [
    {
      "id": "kenney-fps",
      "name": "Starter Kit FPS",
      "category": "fps",
      "description": "...",
      "license": "MIT (code), CC0 (assets)",
      "features": ["..."],
      "godot_version": "4.3+"
    }
  ],
  "categories": ["fps", "2d-platformer", ...],
  "count": 7,
  "filtered_by": "all"
}
```

### Install Template

```json
{
  "op": "templates.install",
  "template_id": "kenney-fps",
  "target_path": "/path/to/extract/template"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Template 'Starter Kit FPS' installed successfully",
  "template_id": "kenney-fps",
  "template_name": "Starter Kit FPS",
  "path": "/path/to/extract/template",
  "entry_scene": "scenes/main.tscn",
  "license": "MIT (code), CC0 (assets)"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Template 'invalid-id' not found in catalog",
  "available_templates": ["kenney-fps", "g2p-2d-platformer", ...]
}
```

## Example AI Agent Workflow

1. **User asks**: "I want to create a first-person shooter game"

2. **AI lists FPS templates**:
   ```json
   {"op": "templates.list", "template_category": "fps"}
   ```

3. **AI installs chosen template**:
   ```json
   {
     "op": "templates.install",
     "template_id": "kenney-fps",
     "target_path": "/Users/username/MyFPSGame"
   }
   ```

4. **AI confirms**: "I've installed the Kenney FPS Starter Kit to your project directory. It includes a character controller, weapon switching, enemies, and CC0 assets. The main scene is at `scenes/main.tscn`."

## Testing

Run the test suite:

```bash
cd backend
python3 test_template_manager.py
```

For full tests including actual download:
```bash
python3 test_template_manager.py --full
```

## Implementation Files

- **`game_templates_catalog.json`**: Catalog of all available templates with metadata
- **`template_manager.py`**: Core module for listing and installing templates
- **`Godot_tools.py`**: Updated with `templates.list` and `templates.install` operations
- **`app.py`**: Wired up template operations in `project_manager_internal()`
- **`test_template_manager.py`**: Test suite for validation

## Adding New Templates

To add a new template to the catalog, edit `game_templates_catalog.json`:

```json
{
  "id": "my-template-id",
  "name": "My Game Template",
  "category": "fps",
  "engine": "godot-4",
  "description": "Description of the template",
  "source": {
    "type": "github-zip",
    "url": "https://github.com/user/repo/archive/refs/heads/main.zip",
    "repo_url": "https://github.com/user/repo"
  },
  "license": "MIT",
  "entry_scene": "main.tscn",
  "features": ["feature1", "feature2"],
  "godot_version": "4.0+"
}
```

**Requirements:**
- Template must be open-source (MIT, CC0, Unlicense preferred)
- Must be compatible with Godot 4.x
- Must include a valid `project.godot` file
- GitHub archive URL should point to a stable branch or tag

## License Compliance

All templates in the catalog are:
- **MIT**, **CC0**, or **Unlicense** licensed (free for any use)
- Properly attributed in the catalog
- From reputable sources (Kenney, GDQuest, verified community creators)

When using templates, the agent should inform users of the license terms from the response.

