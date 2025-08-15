## Orca Engine

Based on Godot Engine, with enhancements by Simplifine.

### Overview
Describe what Orca adds and who it is for.

### Whats the catch?
We are integrating a chat bot, with complete access to Godot. 
The chatbot can:
  - Read/edit/craete/delete files
  - understand the entire project as context
  - create images and keep consistency across images created
  - edit godot native objects, e.g. nodes, scenes, ...

### Quick start
macOS example:
```bash
brew install scons pkg-config
python3 --version   # Python 3.8+

# Build editor (replace platform as needed)
scons -j$(sysctl -n hw.ncpu) platform=macos
bin/godot.osx.editor.universal
```

See upstream Godot docs for full build/platform details.

### License
- Upstream Godot Engine code: Expat (MIT). See `LICENSE.txt`.
- Third-party components: see `COPYRIGHT.txt` and licenses under `thirdparty/`.
- Simplifine original contributions: Non-commercial source-available. See `NOTICE` and `LICENSES/COMPANY-NONCOMMERCIAL.md`.

Commercial licensing is available. Contact: [contact-email-or-url]

### Attribution
This project is based on Godot Engine by the Godot Engine contributors, Juan Linietsky and Ariel Manzur. We are not affiliated with the Godot project.

### Branding
This project is an independent distribution by Simplifine. “Godot” and related marks are property of their respective owners.


