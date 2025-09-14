#!/bin/bash

set -euo pipefail

# Validation script for Orca Engine Linux Build System
# This script validates that all components are properly installed and configured

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔍 Validating Orca Engine Linux Build System"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

info() {
    echo -e "ℹ️  $1"
}

# Change to project root
cd "$PROJECT_ROOT"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Check required files exist
echo "🔍 Checking required files..."

required_files=(
    "build_linux_orca.sh"
    "docker/Dockerfile.orca-builder"
    "docker/Dockerfile.orca-builder-production"
    "scripts/package-linux-orca.sh"
    "scripts/sign-linux-binary.sh"
    "scripts/release-orca.sh"
    ".github/workflows/linux-production.yml"
    ".github/workflows/release-automation.yml"
    "docs/LINUX_BUILD_GUIDE.md"
    "LINUX_BUILD_SUMMARY.md"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        success "Found: $file"
    else
        error "Missing: $file"
        ((missing_files++))
    fi
done

if [[ $missing_files -gt 0 ]]; then
    error "$missing_files required files are missing!"
    exit 1
fi

echo ""

# Check file permissions
echo "🔍 Checking file permissions..."

executable_files=(
    "build_linux_orca.sh"
    "scripts/package-linux-orca.sh"
    "scripts/sign-linux-binary.sh"
    "scripts/release-orca.sh"
)

permission_issues=0
for file in "${executable_files[@]}"; do
    if [[ -x "$file" ]]; then
        success "Executable: $file"
    else
        warning "Not executable: $file (fixing...)"
        chmod +x "$file"
        if [[ -x "$file" ]]; then
            success "Fixed: $file"
        else
            error "Failed to fix: $file"
            ((permission_issues++))
        fi
    fi
done

if [[ $permission_issues -gt 0 ]]; then
    error "$permission_issues permission issues could not be fixed!"
    exit 1
fi

echo ""

# Test script help outputs
echo "🔍 Testing script help outputs..."

scripts_to_test=(
    "./build_linux_orca.sh --help"
    "./scripts/package-linux-orca.sh --help"
    "./scripts/sign-linux-binary.sh --help"
    "./scripts/release-orca.sh --help"
)

for script_cmd in "${scripts_to_test[@]}"; do
    if $script_cmd >/dev/null 2>&1; then
        success "Help works: $script_cmd"
    else
        error "Help failed: $script_cmd"
        exit 1
    fi
done

echo ""

# Check Docker availability
echo "🔍 Checking Docker availability..."

if command -v docker >/dev/null 2>&1; then
    if docker info >/dev/null 2>&1; then
        success "Docker is available and running"
    else
        warning "Docker is installed but not running"
        info "Docker daemon may need to be started for builds to work"
    fi
else
    warning "Docker is not installed"
    info "Docker is required for containerized builds"
fi

echo ""

# Check git repository
echo "🔍 Checking git repository..."

if [[ -d ".git" ]]; then
    success "Git repository detected"
    
    # Check if we can get current branch
    if git branch --show-current >/dev/null 2>&1; then
        current_branch=$(git branch --show-current)
        info "Current branch: $current_branch"
    fi
    
    # Check if we have any tags
    if git tag -l | head -1 >/dev/null 2>&1; then
        tag_count=$(git tag -l | wc -l)
        info "Git tags available: $tag_count"
    else
        info "No git tags found (normal for new repositories)"
    fi
else
    warning "Not a git repository"
    info "Some features like version detection may not work properly"
fi

echo ""

# Validate GitHub Actions workflows
echo "🔍 Validating GitHub Actions workflows..."

workflow_files=(
    ".github/workflows/linux-production.yml"
    ".github/workflows/release-automation.yml"
)

for workflow in "${workflow_files[@]}"; do
    if [[ -f "$workflow" ]]; then
        # Basic YAML validation
        if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
            success "Valid YAML: $workflow"
        else
            error "Invalid YAML: $workflow"
            exit 1
        fi
    fi
done

echo ""

# Check for SCons build system
echo "🔍 Checking build system..."

if [[ -f "SConstruct" ]]; then
    success "SCons build file found"
else
    error "SConstruct not found - this may not be a valid Orca/Godot project"
    exit 1
fi

# Check for platform detection
if [[ -d "platform/linuxbsd" ]]; then
    success "Linux platform support found"
else
    error "Linux platform support not found"
    exit 1
fi

echo ""

# Validate Dockerfile syntax
echo "🔍 Validating Dockerfile syntax..."

dockerfiles=(
    "docker/Dockerfile.orca-builder"
    "docker/Dockerfile.orca-builder-production"
)

for dockerfile in "${dockerfiles[@]}"; do
    if [[ -f "$dockerfile" ]]; then
        # Basic Dockerfile validation
        if grep -q "^FROM " "$dockerfile" && grep -q "^RUN " "$dockerfile"; then
            success "Valid Dockerfile: $dockerfile"
        else
            error "Invalid Dockerfile: $dockerfile"
            exit 1
        fi
    fi
done

echo ""

# Check documentation
echo "🔍 Checking documentation..."

doc_files=(
    "docs/LINUX_BUILD_GUIDE.md"
    "LINUX_BUILD_SUMMARY.md"
)

for doc in "${doc_files[@]}"; do
    if [[ -f "$doc" && -s "$doc" ]]; then
        line_count=$(wc -l < "$doc")
        success "Documentation: $doc ($line_count lines)"
    else
        error "Documentation missing or empty: $doc"
        exit 1
    fi
done

echo ""

# Summary
echo "📊 Validation Summary"
echo "===================="
echo ""

success "All required files are present"
success "All scripts have proper permissions"
success "All scripts have working help output"
success "GitHub Actions workflows are valid YAML"
success "Build system files are present"
success "Documentation is complete"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    success "Docker is ready for builds"
else
    warning "Docker setup may need attention"
fi

echo ""
echo "🎉 Linux Build System Validation Complete!"
echo ""
echo "Next steps:"
echo "1. 🧪 Test a development build: ./build_linux_orca.sh"
echo "2. 🚀 Test a production build: ./build_linux_orca.sh --production"
echo "3. 📦 Test packaging: ./scripts/package-linux-orca.sh --binary <path> --all"
echo "4. 🏷️  Create a test release: ./scripts/release-orca.sh --version v0.1.0-test --dry-run"
echo ""
echo "For detailed instructions, see: docs/LINUX_BUILD_GUIDE.md"