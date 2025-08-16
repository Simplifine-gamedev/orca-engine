## Orca Engine

Discord: https://discord.gg/bvdpdT26Tq

Based on Godot Engine, with enhancements by Simplifine.


### What's the catch?
We are integrating a chat bot, with complete access to Godot. 
The chatbot can:
  - Read/edit/create/delete files
  - understand the entire project as context
  - create images and keep consistency across images created
  - edit godot native objects, e.g. nodes, scenes, ...

### AI Assistant Capabilities

The integrated chatbot has access to comprehensive tools for Godot development:

#### Scene & Node Management
- **Scene Operations**: Open, create, save, and instantiate scenes
- **Node Manipulation**: Create, delete, move, and rename nodes in the scene tree
- **Node Inspection**: Get node properties, scripts, and hierarchical information
- **Node Selection**: Work with currently selected nodes in the editor
- **Class Discovery**: Access all available Godot node classes and their capabilities

#### Script & Code Management
- **File Operations**: Read, write, and edit any project file with line-specific targeting
- **Script Generation**: Create new GDScript files with AI-powered code generation
- **Code Editing**: Apply intelligent edits to existing scripts with natural language prompts
- **Error Detection**: Check for compilation errors across single files or entire project
- **Script Attachment**: Attach and manage scripts on nodes

#### Project Navigation & Search
- **File System**: Browse project directories and files with filtering options
- **Semantic Search**: Find relevant files and code using natural language queries
- **Project Context**: Understand relationships between files and project structure
- **Graph Analysis**: Discover connected files and central project components

#### Visual Content Creation
- **Image Generation**: Create new images from text descriptions with various art styles
- **Image Editing**: Modify existing images using AI-powered editing
- **Asset Management**: Save generated images to specific project paths
- **Style Consistency**: Maintain visual coherence across generated assets

#### Physics & Game Objects
- **Collision Shapes**: Add collision detection to physics bodies (rectangle, circle, capsule)
- **Node Properties**: Modify transform, physics, and gameplay properties
- **Method Calls**: Execute node methods with custom parameters

#### Signal & Connection System
- **Signal Inspection**: List available signals and their connections
- **Connection Management**: Create and manage signal connections between nodes
- **Signal Tracing**: Debug signal flow with lightweight event tracking
- **Connection Validation**: Verify signal connection integrity

#### Development Workflow
- **Multi-Model Support**: Choose between GPT-5, Claude-4, Gemini-2.5, and GPT-4o
- **Real-time Assistance**: Stream responses with tool execution feedback
- **Error Recovery**: Intelligent fallback between AI providers for reliability
- **Authentication**: Secure access with OAuth (Google, GitHub, Microsoft) or guest mode

### Build & Setup Instructions

Orca Engine requires building both the Godot editor and setting up the AI backend server.

#### Prerequisites

**All Platforms:**
- Python 3.8+ (for AI backend)
- Git (for cloning repository)

#### macOS Setup

```bash
# Install build dependencies
brew install scons pkg-config python3

# Verify Python version
python3 --version   # Should be 3.8+

# Clone and build Godot editor
git clone https://github.com/yourusername/orca-engine.git
cd godot
scons platform=macos target=editor dev_build=yes -j$(sysctl -n hw.ncpu)

# Setup AI backend
cd backend
python3 -m pip install -r requirements.txt

# Create environment file (copy and modify as needed)
cp .env.example .env   # Configure your API keys

# Start backend server
python3 app.py

# In another terminal, run the editor
cd ..
./bin/godot.macos.editor.dev.arm64
```

#### Windows Setup

```bash
# Install dependencies (using scoop or chocolatey)
# With Scoop:
scoop install python scons git
# Or with Chocolatey:
# choco install python scons git

# Verify Python version
python --version   # Should be 3.8+

# Clone and build Godot editor
git clone https://github.com/yourusername/orca-engine.git
cd godot
scons platform=windows target=editor dev_build=yes -j%NUMBER_OF_PROCESSORS%

# Setup AI backend
cd backend
python -m pip install -r requirements.txt

# Create environment file
copy .env.example .env   # Configure your API keys in a text editor

# Start backend server
python app.py

# In another terminal, run the editor
cd ..
bin\godot.windows.editor.dev.x86_64.exe
```

#### Linux Setup (Ubuntu/Debian)

```bash
# Install build dependencies
sudo apt update
sudo apt install build-essential scons pkg-config libx11-dev libxcursor-dev \
    libxinerama-dev libgl1-mesa-dev libglu1-mesa-dev libasound2-dev \
    libpulse-dev libudev-dev libxi-dev libxrandr-dev python3 python3-pip

# Verify Python version
python3 --version   # Should be 3.8+

# Clone and build Godot editor
git clone https://github.com/yourusername/orca-engine.git
cd godot
scons platform=linuxbsd target=editor dev_build=yes -j$(nproc)

# Setup AI backend
cd backend
python3 -m pip install -r requirements.txt

# Create environment file
cp .env.example .env   # Configure your API keys

# Start backend server
python3 app.py

# In another terminal, run the editor
cd ..
./bin/godot.linuxbsd.editor.dev.x86_64
```

#### Linux Setup (Arch/Manjaro)

```bash
# Install build dependencies
sudo pacman -S base-devel scons pkgconf libx11 libxcursor libxinerama \
    mesa glu alsa-lib pulseaudio systemd libxi libxrandr python python-pip

# Follow the same build steps as Ubuntu above
```

#### Environment Configuration

Create a `.env` file in the `backend/` directory with your API keys:

```env
# Required: At least one AI provider
OPENAI_API_KEY=sk-proj-abc123...
# ANTHROPIC_API_KEY=sk-ant-api03-abc123...
# GOOGLE_API_KEY=AIzaSyAbc123...

# Optional: For advanced features
GCP_PROJECT_ID=my-project-id
FLASK_SECRET_KEY=a-secure-random-string-here

# Development mode (set to 'false' in production)
DEV_MODE=true
```

#### Indexing performance (large projects)

The editor triggers server-side indexing automatically. For large codebases, set these to speed things up (in `backend/.env` locally or before running `backend/deploy.sh` so Cloud Run picks them up as secrets):

```env
# Parallelism and batching
INDEX_MAX_WORKERS=32         # number of parallel file workers on the backend
EMBED_MAX_PARALLEL=12        # concurrent embedding batches (respect provider limits)
EMBED_BATCH_SIZE=256         # embeddings per batch
CHUNK_MAX_LINES=100          # larger chunks = fewer embedding calls

# Small response caches
SEARCH_CACHE_TTL=45
GRAPH_CACHE_TTL=45
```

Notes:
- You can also pass `INDEX_MAX_WORKERS` from the editor via env; it is forwarded to the backend for `index_project`/`index_files`.
- When deploying to Cloud Run, ensure sufficient resources (e.g., `--cpu 4`, `--memory 8Gi`, higher `--concurrency`).

### Godot Docs Search (RAG)

To index and query the official Godot documentation corpus used by the `search_across_godot_docs` tool, see:

- [Godot Docs Search: Indexing and Querying](./godot_doc_search.md)

#### Quick Start Commands

After setup, use these commands to run Orca Engine:

```bash
# Terminal 1: Start AI backend
cd backend && python3 app.py

# Terminal 2: Start Orca Engine editor
# Use the appropriate binary for your platform:
# macOS: ./bin/godot.macos.editor.dev.arm64
# Windows: bin\godot.windows.editor.dev.x86_64.exe  
# Linux: ./bin/godot.linuxbsd.editor.dev.x86_64
```

#### Troubleshooting

- **Build errors**: See [upstream Godot build documentation](https://docs.godotengine.org/en/stable/development/compiling/index.html) for platform-specific issues
- **Python dependency issues**: Ensure you're using Python 3.8+ and consider using a virtual environment
- **AI backend connection**: Verify the backend is running on http://localhost:8000 and API keys are properly configured
- **Missing tools**: The chatbot tools will only appear once the backend connection is established

### License
- Upstream Godot Engine code: Expat (MIT). See `LICENSE.txt`.
- Third-party components: see `COPYRIGHT.txt` and licenses under `thirdparty/`.
- Simplifine original contributions: Non-commercial source-available. See `NOTICE` and `LICENSES/COMPANY-NONCOMMERCIAL.md`.

Commercial licensing is available. Contact: [support@simplifine.com]

### Attribution
This project is based on Godot Engine by the Godot Engine contributors, Juan Linietsky and Ariel Manzur. We are not affiliated with the Godot project.

### Branding
This project is an independent distribution by Simplifine. “Godot” and related marks are property of their respective owners.


