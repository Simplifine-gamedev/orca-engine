#!/bin/bash

set -euo pipefail

# Orca Engine Release Management Script
# This script helps manage releases for Orca Engine

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=""
PRERELEASE=false
DRY_RUN=false
FORCE=false

usage() {
    echo "Usage: $0 --version <version> [options]"
    echo ""
    echo "Required:"
    echo "  --version     Version to release (e.g., v1.0.0, 1.2.3)"
    echo ""
    echo "Options:"
    echo "  --prerelease  Mark as pre-release"
    echo "  --dry-run     Show what would be done without making changes"
    echo "  --force       Force release even if checks fail"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --version v1.0.0"
    echo "  $0 --version v1.1.0-beta --prerelease"
    echo "  $0 --version v1.0.1 --dry-run"
    exit 0
}

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

info() {
    echo "🚀 $1"
}

warning() {
    echo "⚠️  $1"
}

error() {
    echo "❌ ERROR: $1" >&2
    exit 1
}

success() {
    echo "✅ $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --prerelease)
            PRERELEASE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate required parameters
if [[ -z "$VERSION" ]]; then
    error "Version is required (--version)"
fi

# Normalize version (add 'v' prefix if not present)
if [[ ! "$VERSION" =~ ^v ]]; then
    VERSION="v$VERSION"
fi

info "Preparing release: $VERSION"
if [[ "$PRERELEASE" == true ]]; then
    info "This will be marked as a pre-release"
fi

if [[ "$DRY_RUN" == true ]]; then
    warning "DRY RUN MODE - No changes will be made"
fi

# Change to repository root
cd "$SCRIPT_DIR/.."

# Verify we're in a git repository
if [[ ! -d ".git" ]]; then
    error "Not in a git repository"
fi

# Check if we're on main/master branch
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "master" && "$FORCE" != true ]]; then
    error "Not on main/master branch. Use --force to override or switch to main branch."
fi

# Check for uncommitted changes
if [[ -n "$(git status --porcelain)" && "$FORCE" != true ]]; then
    error "Uncommitted changes detected. Commit or stash changes first, or use --force to override."
fi

# Check if tag already exists
if git rev-parse "$VERSION" >/dev/null 2>&1; then
    if [[ "$FORCE" != true ]]; then
        error "Tag $VERSION already exists. Use --force to override."
    else
        warning "Tag $VERSION already exists but --force specified"
    fi
fi

# Fetch latest changes
log "Fetching latest changes..."
if [[ "$DRY_RUN" != true ]]; then
    git fetch origin
fi

# Check if we're up to date with remote
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse "origin/$CURRENT_BRANCH" 2>/dev/null || echo "")

if [[ -n "$REMOTE_COMMIT" && "$LOCAL_COMMIT" != "$REMOTE_COMMIT" && "$FORCE" != true ]]; then
    error "Local branch is not up to date with remote. Pull latest changes or use --force to override."
fi

# Validate version format
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    warning "Version format may not be standard. Expected format: v1.2.3 or v1.2.3-beta"
    if [[ "$FORCE" != true ]]; then
        error "Use --force to override version format validation"
    fi
fi

# Show release summary
echo ""
info "Release Summary:"
echo "  Version: $VERSION"
echo "  Branch: $CURRENT_BRANCH"
echo "  Commit: $LOCAL_COMMIT"
echo "  Pre-release: $PRERELEASE"
echo "  Dry run: $DRY_RUN"
echo ""

# Ask for confirmation if not dry run
if [[ "$DRY_RUN" != true ]]; then
    read -p "❓ Continue with release? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Release cancelled."
        exit 0
    fi
fi

# Create and push tag
log "Creating tag $VERSION..."
if [[ "$DRY_RUN" != true ]]; then
    git tag -a "$VERSION" -m "Release $VERSION"
    git push origin "$VERSION"
    success "Tag $VERSION created and pushed"
else
    echo "DRY RUN: Would create and push tag $VERSION"
fi

# Trigger release workflow
log "Triggering release workflow..."
if [[ "$DRY_RUN" != true ]]; then
    # The tag push will automatically trigger the release workflow
    success "Release workflow triggered by tag push"
    
    echo ""
    info "Release process started!"
    echo "🔗 Monitor progress at: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git/\1/')/actions"
    echo "📦 Release will be available at: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^/]*\/[^/]*\)\.git/\1/')/releases/tag/$VERSION"
    
    # Wait a moment and check if workflow started
    sleep 5
    echo ""
    info "You can also monitor the build progress with GitHub CLI:"
    echo "  gh run list --workflow=release-automation.yml"
    echo "  gh run watch"
else
    echo "DRY RUN: Would trigger release workflow for $VERSION"
fi

echo ""
success "Release script completed!"

if [[ "$DRY_RUN" != true ]]; then
    echo ""
    info "Next steps:"
    echo "1. 📊 Monitor the GitHub Actions workflow"
    echo "2. 📝 Review the generated release notes"
    echo "3. 🧪 Test the released packages"
    echo "4. 📢 Announce the release to the community"
    echo ""
    echo "The release process is now automated and will:"
    echo "- ✅ Build Linux binaries (production optimized)"
    echo "- ✅ Create AppImage, DEB, and TAR.GZ packages"
    echo "- ✅ Sign binaries and generate checksums"
    echo "- ✅ Upload all assets to GitHub Releases"
    echo "- ✅ Generate release notes"
    echo "- ✅ Create community discussion post"
fi