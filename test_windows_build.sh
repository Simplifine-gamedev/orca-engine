#!/bin/bash

# Test script to verify Windows build workflow components
# This simulates the key steps that might fail in the Windows CI

echo "🧪 Testing Windows Build Components"
echo "=================================="

# Test 1: Check if jar directory and CodeSignTool exist
echo "1. Checking CodeSignTool setup..."
if [ -d "jar" ]; then
    echo "✅ jar/ directory exists"
    if [ -f "jar/code_sign_tool-1.3.2.jar" ]; then
        echo "✅ CodeSignTool JAR found"
        echo "   Size: $(stat -f%z jar/code_sign_tool-1.3.2.jar 2>/dev/null || echo 'unknown') bytes"
    else
        echo "❌ CodeSignTool JAR missing: jar/code_sign_tool-1.3.2.jar"
        echo "   Contents of jar/:"
        ls -la jar/ 2>/dev/null || echo "   Directory is empty or doesn't exist"
    fi
else
    echo "❌ jar/ directory missing"
fi

# Test 2: Check if Java is available (simulating CI environment)
echo ""
echo "2. Checking Java availability..."
if command -v java &> /dev/null; then
    echo "✅ Java is available"
    java -version 2>&1 | head -1
else
    echo "❌ Java not found in PATH"
fi

# Test 3: Check if CodeSignTool scripts exist
echo ""
echo "3. Checking CodeSignTool scripts..."
if [ -f "CodeSignTool.sh" ]; then
    echo "✅ CodeSignTool.sh exists"
    if [ -x "CodeSignTool.sh" ]; then
        echo "✅ CodeSignTool.sh is executable"
    else
        echo "⚠️  CodeSignTool.sh is not executable"
    fi
else
    echo "❌ CodeSignTool.sh missing"
fi

if [ -f "CodeSignTool.bat" ]; then
    echo "✅ CodeSignTool.bat exists"
else
    echo "❌ CodeSignTool.bat missing"
fi

# Test 4: Check certificate files
echo ""
echo "4. Checking certificate files..."
if [ -f "eSigner_CKA.zip" ]; then
    echo "✅ Certificate zip found"
    echo "   Size: $(stat -f%z eSigner_CKA.zip 2>/dev/null || echo 'unknown') bytes"
else
    echo "❌ Certificate zip missing: eSigner_CKA.zip"
fi

# Test 5: Check if bin directory structure is ready
echo ""
echo "5. Checking bin directory structure..."
if [ -d "bin" ]; then
    echo "✅ bin/ directory exists"
    echo "   Contents:"
    ls -la bin/ | head -10
    
    # Look for existing Windows binaries
    if ls bin/*windows*.exe &> /dev/null; then
        echo "✅ Found existing Windows binaries:"
        ls -la bin/*windows*.exe
    else
        echo "ℹ️  No Windows binaries found (this is normal if not built yet)"
    fi
else
    echo "❌ bin/ directory missing"
fi

# Test 6: Simulate the workflow steps
echo ""
echo "6. Simulating workflow steps..."

echo "   Step 1: Create signed directory..."
mkdir -p bin/signed
echo "   ✅ bin/signed created"

echo "   Step 2: Test JAR execution (dry run)..."
if [ -f "jar/code_sign_tool-1.3.2.jar" ] && command -v java &> /dev/null; then
    # Test if the JAR can be executed (just check if it responds to help)
    if java -jar jar/code_sign_tool-1.3.2.jar --help &>/dev/null; then
        echo "   ✅ CodeSignTool JAR is executable"
    else
        echo "   ⚠️  CodeSignTool JAR may have issues (or requires specific args)"
    fi
else
    echo "   ⚠️  Cannot test JAR execution (missing Java or JAR)"
fi

echo ""
echo "🎯 Test Summary:"
echo "==============="

# Count issues
issues=0

[ ! -d "jar" ] && ((issues++))
[ ! -f "jar/code_sign_tool-1.3.2.jar" ] && ((issues++))
[ ! -f "CodeSignTool.sh" ] && ((issues++))
[ ! -f "CodeSignTool.bat" ] && ((issues++))
[ ! -f "eSigner_CKA.zip" ] && ((issues++))

if [ $issues -eq 0 ]; then
    echo "✅ All components ready for Windows build"
    echo "   The workflow should work if secrets are properly configured"
else
    echo "❌ Found $issues issue(s) that may cause Windows build to fail"
    echo "   Please fix the missing components listed above"
fi

echo ""
echo "💡 Next steps:"
echo "   1. Ensure all missing files are added to the repository"
echo "   2. Verify GitHub secrets are configured: SSL_COM_USERNAME, SSL_COM_PASSWORD, SSL_COM_TOTP_SECRET"
echo "   3. Test the workflow with a push to main branch"
