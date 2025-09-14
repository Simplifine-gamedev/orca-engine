# Orca Engine Linux Build Guide

This guide covers building, packaging, and distributing Orca Engine for Linux systems.

## Table of Contents

- [Quick Start](#quick-start)
- [Build System Overview](#build-system-overview)
- [Development Builds](#development-builds)
- [Production Builds](#production-builds)
- [Packaging](#packaging)
- [Binary Signing](#binary-signing)
- [Automated Releases](#automated-releases)
- [Cross-Distribution Testing](#cross-distribution-testing)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Development Build

For development and testing:

```bash
# Simple development build
./build_linux_orca.sh

# The binary will be available at: bin/godot.linuxbsd.editor.x86_64
```

### Production Build

For distribution and releases:

```bash
# Production build with optimizations
./build_linux_orca.sh --production

# Creates optimized binary with static linking and compression
```

## Build System Overview

The Orca Engine Linux build system consists of:

- **SCons**: Primary build system (inherited from Godot)
- **Docker**: Containerized build environment for consistency
- **GitHub Actions**: Automated CI/CD pipeline
- **Custom Scripts**: Additional tooling for packaging and distribution

### Key Files

- `build_linux_orca.sh` - Main build script
- `docker/Dockerfile.orca-builder` - Development build environment
- `docker/Dockerfile.orca-builder-production` - Production build environment
- `.github/workflows/linux-production.yml` - CI/CD pipeline
- `scripts/` - Additional tooling

## Development Builds

Development builds are optimized for fast compilation and debugging.

### Local Development Build

```bash
# Clone the repository
git clone <repository-url>
cd orca-engine

# Run development build
./build_linux_orca.sh

# Test the binary
./bin/godot.linuxbsd.editor.x86_64 --version
```

### Build Configuration

Development builds use these settings:
- `production=no` - Development mode
- `use_lto=no` - Faster linking
- `debug_symbols=no` - Smaller binaries
- `optimize=speed` - Speed optimization

### Docker Environment

The build runs in a Ubuntu 22.04 container with all dependencies pre-installed:

```dockerfile
FROM ubuntu:22.04
# Includes: build-essential, scons, pkg-config, graphics libraries, etc.
```

## Production Builds

Production builds are optimized for distribution with maximum compatibility.

### Production Build Command

```bash
./build_linux_orca.sh --production
```

### Production Configuration

Production builds use these optimizations:
- `production=yes` - Production mode with all optimizations
- `use_lto=auto` - Link-time optimization
- `use_static_cpp=yes` - Static C++ runtime linking
- `builtin_*=yes` - Embed all dependencies
- Binary stripping and UPX compression

### Build Output

Production builds create:
- `bin/godot.linuxbsd.editor.x86_64` - Main binary
- `bin/orca-engine-linux-x86_64` - Branded binary name

## Packaging

The packaging system creates multiple distribution formats.

### Using the Packaging Script

```bash
# Create all package types
./scripts/package-linux-orca.sh --binary bin/godot.linuxbsd.editor.x86_64 --all --version v1.0.0

# Create specific package types
./scripts/package-linux-orca.sh --binary bin/godot.linuxbsd.editor.x86_64 --appimage --version v1.0.0
./scripts/package-linux-orca.sh --binary bin/godot.linuxbsd.editor.x86_64 --deb --version v1.0.0
./scripts/package-linux-orca.sh --binary bin/godot.linuxbsd.editor.x86_64 --tar --version v1.0.0
```

### Package Types

#### 1. TAR.GZ Package (Recommended)

- **Format**: `orca-engine-VERSION-linux-x86_64.tar.gz`
- **Contents**: Binary, launcher script, documentation
- **Use case**: General distribution, works on all Linux systems

#### 2. AppImage

- **Format**: `OrcaEngine-VERSION-x86_64.AppImage`
- **Contents**: Self-contained portable application
- **Use case**: Portable installation, no system integration needed

#### 3. DEB Package

- **Format**: `orca-engine_VERSION-1_amd64.deb`
- **Contents**: Debian/Ubuntu package with system integration
- **Use case**: Debian, Ubuntu, and derivatives

### Package Contents

Each package includes:
- Orca Engine binary
- Launcher script
- Desktop integration files
- Documentation (README, LICENSE)
- Icon files

## Binary Signing

Binary signing provides integrity verification and authenticity.

### Using the Signing Script

```bash
# Sign binary with checksums only
./scripts/sign-linux-binary.sh --binary bin/godot.linuxbsd.editor.x86_64 --output signed/

# Sign with GPG (requires GPG setup)
./scripts/sign-linux-binary.sh --binary bin/godot.linuxbsd.editor.x86_64 --output signed/ --key YOUR_GPG_KEY_ID
```

### Generated Files

The signing process creates:
- `binary.sha256` - SHA256 checksum
- `binary.sha512` - SHA512 checksum
- `binary.md5` - MD5 checksum
- `binary.checksums` - Combined checksums file
- `verify.sh` - Verification script
- `binary.sig` - GPG signature (if available)
- `binary.checksums.asc` - GPG signed checksums (if available)

### Verification

Users can verify downloads:

```bash
# Extract signed package
tar -xzf orca-engine-linux-x86_64-v1.0.0-signed.tar.gz

# Run verification
./verify.sh
```

## Automated Releases

The automated release system handles the complete release process.

### GitHub Actions Workflow

The `linux-production.yml` workflow:

1. **Build**: Creates production binary
2. **Package**: Generates all package formats
3. **Sign**: Creates checksums and signatures
4. **Test**: Validates on multiple Ubuntu versions
5. **Upload**: Publishes to GitHub Releases

### Triggering Releases

#### Method 1: Release Script

```bash
# Create and push release tag
./scripts/release-orca.sh --version v1.0.0

# For pre-release
./scripts/release-orca.sh --version v1.0.0-beta --prerelease

# Dry run to test
./scripts/release-orca.sh --version v1.0.0 --dry-run
```

#### Method 2: Manual Tag

```bash
# Create and push tag manually
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

#### Method 3: GitHub UI

Use the GitHub web interface to create a release, which triggers the workflow.

### Release Artifacts

Each release includes:
- `orca-engine-linux-x86_64-VERSION.tar.gz` - Main distribution package
- `orca-engine-linux-x86_64-VERSION-signed.tar.gz` - Signed package with checksums
- `OrcaEngine-VERSION-x86_64.AppImage` - AppImage package
- `orca-engine_VERSION-1_amd64.deb` - DEB package

## Cross-Distribution Testing

The CI/CD pipeline tests compatibility across Linux distributions.

### Tested Distributions

- Ubuntu 20.04 LTS
- Ubuntu 22.04 LTS
- Additional distributions can be added to the matrix

### Local Testing

Test on different distributions using Docker:

```bash
# Test on Ubuntu 20.04
docker run --rm -v $(pwd)/bin:/app ubuntu:20.04 /app/godot.linuxbsd.editor.x86_64 --version

# Test on different distributions
docker run --rm -v $(pwd)/bin:/app debian:bullseye /app/godot.linuxbsd.editor.x86_64 --version
docker run --rm -v $(pwd)/bin:/app fedora:latest /app/godot.linuxbsd.editor.x86_64 --version
```

## System Requirements

### Build Requirements

- Docker (for containerized builds)
- Git
- Bash
- 8GB+ RAM (for building)
- 10GB+ free disk space

### Runtime Requirements (for built binary)

- Linux x86_64
- OpenGL 3.3 compatible graphics card
- 2GB RAM minimum, 4GB recommended
- 500MB free disk space
- Audio system (ALSA/PulseAudio)

### Dependencies

The production build statically links most dependencies, but may still require:
- `libc6` (glibc)
- `libstdc++6`
- `libgl1-mesa-glx` or equivalent OpenGL library
- `libasound2` or PulseAudio

## Troubleshooting

### Build Issues

#### Docker Build Fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -f docker/Dockerfile.orca-builder-production -t orca-builder-production .
```

#### SCons Build Fails

Check the SCons log for specific errors:
- Missing dependencies
- Compiler issues
- Memory limitations

#### Binary Size Too Large

The production build includes compression:
- UPX compression (if available)
- Strip debugging symbols
- Static linking reduces external dependencies

### Runtime Issues

#### Binary Won't Start

```bash
# Check dependencies
ldd bin/godot.linuxbsd.editor.x86_64

# Check if executable
file bin/godot.linuxbsd.editor.x86_64

# Test basic functionality
./bin/godot.linuxbsd.editor.x86_64 --version
./bin/godot.linuxbsd.editor.x86_64 --help
```

#### Graphics Issues

- Ensure OpenGL 3.3 support: `glxinfo | grep "OpenGL version"`
- Update graphics drivers
- Try software rendering: `LIBGL_ALWAYS_SOFTWARE=1 ./orca-engine`

#### Audio Issues

- Check audio system: `pulseaudio --check` or `aplay -l`
- Test with different audio drivers in Orca Engine settings

### Package Issues

#### AppImage Won't Run

```bash
# Make executable
chmod +x OrcaEngine-v1.0.0-x86_64.AppImage

# Check FUSE support
./OrcaEngine-v1.0.0-x86_64.AppImage --appimage-extract-and-run
```

#### DEB Installation Fails

```bash
# Check dependencies
sudo apt-get install -f

# Force installation
sudo dpkg -i --force-depends orca-engine_1.0.0-1_amd64.deb
```

## Development Workflow

### Making Changes

1. Make code changes
2. Test with development build: `./build_linux_orca.sh`
3. Test production build: `./build_linux_orca.sh --production`
4. Create packages: `./scripts/package-linux-orca.sh --binary bin/godot.linuxbsd.editor.x86_64 --all`
5. Test packages on different systems

### Release Process

1. Ensure all changes are committed
2. Update version information
3. Create release: `./scripts/release-orca.sh --version vX.Y.Z`
4. Monitor GitHub Actions workflow
5. Test released packages
6. Announce release

## Advanced Configuration

### Custom Build Flags

Modify the Dockerfile or build script to add custom SCons flags:

```bash
# Example: Enable specific modules
scons platform=linuxbsd target=editor arch=x86_64 \
    production=yes \
    module_custom_enabled=yes \
    custom_flag=value
```

### Cross-Compilation

For different architectures, modify the build configuration:

```bash
# ARM64 build (experimental)
scons platform=linuxbsd target=editor arch=arm64 production=yes
```

### Custom Packaging

Create custom package formats by modifying `scripts/package-linux-orca.sh` or creating new scripts.

## Support

For build system issues:

- Check existing [GitHub Issues](https://github.com/your-repo/issues)
- Create new issue with build logs
- Join community discussions

For Orca Engine usage:

- Visit [Documentation](https://orcaengine.ai/docs)
- Join [Community Discord](https://discord.gg/orcaengine)
- Check [FAQ](https://orcaengine.ai/faq)