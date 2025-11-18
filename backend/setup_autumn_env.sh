#!/bin/bash
# Setup script for Autumn API key
# Usage: ./setup_autumn_env.sh YOUR_AUTUMN_SECRET_KEY

if [ -z "$1" ]; then
    echo "❌ Error: Please provide your Autumn secret key"
    echo "Usage: ./setup_autumn_env.sh am_sk_your_key_here"
    exit 1
fi

AUTUMN_KEY="$1"

echo "🔧 Setting up Autumn environment..."

# Add to .env file in backend directory
echo "" >> .env
echo "# Autumn Pricing System" >> .env
echo "AUTUMN_SECRET_KEY=$AUTUMN_KEY" >> .env

echo "✅ Added AUTUMN_SECRET_KEY to .env file"

# Add to zsh config for permanent shell access
if grep -q "AUTUMN_SECRET_KEY" ~/.zshrc; then
    echo "⚠️  AUTUMN_SECRET_KEY already exists in ~/.zshrc, skipping..."
else
    echo "" >> ~/.zshrc
    echo "# Autumn API Key for Orca Engine pricing" >> ~/.zshrc
    echo "export AUTUMN_SECRET_KEY=$AUTUMN_KEY" >> ~/.zshrc
    echo "✅ Added AUTUMN_SECRET_KEY to ~/.zshrc"
fi

# Export for current session
export AUTUMN_SECRET_KEY=$AUTUMN_KEY
echo "✅ Exported AUTUMN_SECRET_KEY for current session"

echo ""
echo "🎉 Setup complete! Your Autumn API key is now configured."
echo ""
echo "To verify, run:"
echo "  echo \$AUTUMN_SECRET_KEY"
echo ""
echo "To apply in new terminal sessions, run:"
echo "  source ~/.zshrc"

