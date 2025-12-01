#!/usr/bin/env python3
"""
Validation script for base_game_v2.json
Tests repository URLs and adds license information.
"""

import json
import sys
import requests
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional
import time

# Mapping from base_game_v2.json names/ids to base_games.json licenses
# This will be populated from base_games.json
LICENSE_MAP = {}


def load_license_map():
    """Load license information from base_games.json"""
    try:
        with open('base_games.json', 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        for entry in entries:
            # Map by repo URL (without .git)
            repo_url = entry.get('repo_or_page', '').rstrip('/')
            if repo_url:
                LICENSE_MAP[repo_url] = entry.get('license', 'MIT')
            
            # Also map by title for fallback
            title = entry.get('title', '')
            if title:
                LICENSE_MAP[title.lower()] = entry.get('license', 'MIT')
    except FileNotFoundError:
        print("⚠️  Warning: base_games.json not found, using default MIT licenses")
    except Exception as e:
        print(f"⚠️  Warning: Could not load license map: {e}")


def get_license_for_entry(entry: Dict) -> str:
    """Get license for an entry, with fallbacks"""
    # Try to match by repo URL
    repo_url = entry.get('repo_url', '').replace('.git', '').rstrip('/')
    if repo_url in LICENSE_MAP:
        return LICENSE_MAP[repo_url]
    
    # Try to match by name
    name = entry.get('name', '').lower()
    if name in LICENSE_MAP:
        return LICENSE_MAP[name]
    
    # Default licenses based on patterns
    repo_lower = repo_url.lower()
    if 'godotengine' in repo_lower:
        return "MIT (Godot demo projects)"
    elif 'kenney' in repo_lower:
        return "MIT (code), CC0 assets"
    elif 'brett' in repo_lower or 'brettchalupa' in repo_lower:
        return "Code CC0; assets CC0/CC-BY"
    else:
        return "MIT"  # Default fallback


def validate_url(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Validate if a URL is accessible.
    Returns (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or not a string"
    
    # Parse URL to check format
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format"
    except Exception as e:
        return False, f"URL parsing error: {str(e)}"
    
    # Remove .git for validation (check the web URL)
    check_url = url.replace('.git', '')
    
    # Check if URL is accessible
    try:
        # Use HEAD request first (more efficient)
        response = requests.head(check_url, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return True, f"OK ({response.status_code})"
        
        # If HEAD fails, try GET
        response = requests.get(check_url, timeout=timeout, allow_redirects=True, stream=True)
        if response.status_code < 400:
            return True, f"OK ({response.status_code})"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except requests.exceptions.TooManyRedirects:
        return False, "Too many redirects"
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_entry(entry: Dict, index: int) -> Dict:
    """
    Validate a single entry from base_game_v2.json
    Returns validation results dictionary
    """
    entry_id = entry.get('id', f'entry_{index}')
    name = entry.get('name', 'Unknown')
    
    results = {
        'id': entry_id,
        'name': name,
        'index': index,
        'url_valid': False,
        'url_error': '',
        'license': '',
        'all_valid': False
    }
    
    # Validate URL
    repo_url = entry.get('repo_url', '')
    if repo_url:
        url_valid, url_error = validate_url(repo_url)
        results['url_valid'] = url_valid
        results['url_error'] = url_error
    else:
        results['url_error'] = "Missing 'repo_url' field"
    
    # Get license
    results['license'] = get_license_for_entry(entry)
    
    # Overall validation
    results['all_valid'] = results['url_valid']
    
    return results


def main():
    """Main validation function"""
    json_file = 'base_game_v2.json'
    
    # Load license map
    load_license_map()
    
    # Read JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {json_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {json_file}: {e}")
        sys.exit(1)
    
    if not isinstance(entries, list):
        print(f"❌ Error: {json_file} should contain a JSON array")
        sys.exit(1)
    
    print(f"🔍 Validating {len(entries)} entries from {json_file}...\n")
    
    results = []
    valid_count = 0
    invalid_urls = []
    
    # Validate each entry
    for i, entry in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] Validating: {entry.get('name', 'Unknown')}...", end=' ', flush=True)
        
        result = validate_entry(entry, i)
        results.append(result)
        
        if result['all_valid']:
            print("✅")
            valid_count += 1
        else:
            print("❌")
            invalid_urls.append(result)
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total entries: {len(entries)}")
    print(f"✅ Fully valid: {valid_count}")
    print(f"❌ Invalid URLs: {len(invalid_urls)}")
    
    # Print invalid URLs
    if invalid_urls:
        print("\n" + "-"*80)
        print("INVALID URLS:")
        print("-"*80)
        for result in invalid_urls:
            print(f"\n  [{result['index']+1}] {result['name']} (ID: {result['id']})")
            print(f"      URL: {entries[result['index']].get('repo_url', 'N/A')}")
            print(f"      Error: {result['url_error']}")
    
    # Print all results with licenses
    print("\n" + "="*80)
    print("DETAILED RESULTS WITH LICENSES")
    print("="*80)
    for result in results:
        status = "✅" if result['all_valid'] else "❌"
        print(f"\n{status} [{result['index']+1}] {result['name']}")
        print(f"   ID: {result['id']}")
        print(f"   URL: {'✅ Valid' if result['url_valid'] else '❌ Invalid'} - {result['url_error']}")
        print(f"   License: {result['license']}")
    
    # Exit code
    if valid_count == len(entries):
        print("\n✅ All entries are valid!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(entries) - valid_count} entry/entries have issues")
        sys.exit(1)


if __name__ == '__main__':
    main()



