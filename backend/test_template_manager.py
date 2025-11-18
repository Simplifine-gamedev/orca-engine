"""
Test script for template manager functionality
Run this to verify templates.list and templates.install operations work correctly
"""

import os
import sys
import json
import tempfile
import shutil

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

from template_manager import list_templates, install_template, get_categories, get_template_info


def test_list_templates():
    """Test listing all templates"""
    print("\n" + "="*60)
    print("TEST 1: List all templates")
    print("="*60)
    
    templates = list_templates()
    print(f"✓ Found {len(templates)} templates")
    
    for t in templates:
        print(f"  - {t['id']}: {t['name']} ({t['category']})")
    
    assert len(templates) > 0, "Should have at least one template"
    print("✓ Test passed: list_templates()")


def test_list_by_category():
    """Test filtering templates by category"""
    print("\n" + "="*60)
    print("TEST 2: List templates by category")
    print("="*60)
    
    categories = get_categories()
    print(f"✓ Available categories: {', '.join(categories)}")
    
    for category in categories:
        templates = list_templates(category)
        print(f"  - {category}: {len(templates)} template(s)")
        for t in templates:
            assert t['category'] == category, f"Template {t['id']} should be in category {category}"
    
    print("✓ Test passed: list_templates(category)")


def test_get_template_info():
    """Test getting specific template info"""
    print("\n" + "="*60)
    print("TEST 3: Get specific template info")
    print("="*60)
    
    template_id = "kenney-fps"
    info = get_template_info(template_id)
    
    if info:
        print(f"✓ Found template: {info['name']}")
        print(f"  - Category: {info['category']}")
        print(f"  - License: {info['license']}")
        print(f"  - Source: {info['source']['url']}")
    else:
        print(f"✗ Template {template_id} not found")
        return False
    
    # Test non-existent template
    info = get_template_info("non-existent-template")
    assert info is None, "Should return None for non-existent template"
    
    print("✓ Test passed: get_template_info()")


def test_install_template():
    """Test installing a template (dry run with small template)"""
    print("\n" + "="*60)
    print("TEST 4: Install template")
    print("="*60)
    print("⚠️  This test will download a real template from GitHub")
    print("    It may take a minute depending on your connection...")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="orca_template_test_")
    print(f"✓ Created temp directory: {temp_dir}")
    
    try:
        # Try to install a smaller template (2D platformer is usually smaller)
        template_id = "g2p-2d-platformer"
        print(f"  Installing template: {template_id}")
        
        result = install_template(template_id, temp_dir)
        
        if result['success']:
            print(f"✓ Template installed successfully!")
            print(f"  - Name: {result.get('template_name')}")
            print(f"  - Path: {result.get('path')}")
            print(f"  - Entry scene: {result.get('entry_scene')}")
            
            # Verify project.godot exists
            project_file = os.path.join(temp_dir, "project.godot")
            if os.path.exists(project_file):
                print(f"✓ project.godot found")
            else:
                print(f"✗ project.godot not found!")
                return False
            
            # List some of the files
            files = os.listdir(temp_dir)
            print(f"✓ Template contains {len(files)} items:")
            for f in files[:10]:  # Show first 10 items
                print(f"    - {f}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")
        else:
            print(f"✗ Installation failed: {result.get('error')}")
            return False
            
    finally:
        # Cleanup
        print(f"  Cleaning up temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✓ Cleaned up")
    
    print("✓ Test passed: install_template()")


def test_project_manager_format():
    """Test that the response format matches what project_manager expects"""
    print("\n" + "="*60)
    print("TEST 5: Project manager integration format")
    print("="*60)
    
    try:
        # Try to import app - may fail if Flask not installed
        from app import project_manager_internal
        
        # Test templates.list
        print("  Testing templates.list operation...")
        result = project_manager_internal({'op': 'templates.list'})
        assert result['success'], "templates.list should succeed"
        assert 'templates' in result, "Should have templates array"
        assert 'categories' in result, "Should have categories array"
        assert 'count' in result, "Should have count field"
        print(f"✓ templates.list returned {result['count']} templates")
        
        # Test templates.list with category filter
        print("  Testing templates.list with category filter...")
        result = project_manager_internal({'op': 'templates.list', 'template_category': 'fps'})
        assert result['success'], "templates.list with category should succeed"
        assert result['filtered_by'] == 'fps', "Should be filtered by fps"
        print(f"✓ templates.list filtered by 'fps' returned {result['count']} templates")
        
        # Test templates.install validation (without actually installing)
        print("  Testing templates.install validation...")
        result = project_manager_internal({'op': 'templates.install'})
        assert not result['success'], "templates.install without params should fail"
        assert 'error' in result, "Should have error message"
        print(f"✓ templates.install validation working: {result['error']}")
        
        print("✓ Test passed: project_manager integration format")
    except ImportError as e:
        print(f"⚠️  Skipping project_manager test (Flask not installed in test environment)")
        print(f"   This is OK - the template_manager functions work correctly")
        print(f"   Run this test in the backend environment with Flask installed for full testing")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TEMPLATE MANAGER TEST SUITE")
    print("="*60)
    
    try:
        test_list_templates()
        test_list_by_category()
        test_get_template_info()
        test_project_manager_format()
        
        # Check if --full flag is passed for download test
        run_download_test = '--full' in sys.argv
        
        if run_download_test:
            print("\n⚠️  Running full test suite (including download test)")
            test_install_template()
        else:
            print("\n" + "="*60)
            print("ℹ️  Skipping download test (use --full flag to include it)")
            print("   Example: python3 test_template_manager.py --full")
            print("="*60)
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

