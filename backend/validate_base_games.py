#!/usr/bin/env python3
"""
Validation script for base_games.json
Tests repository URLs and license information for each entry.
"""

import json
import sys
import requests
from urllib.parse import urlparse
from typing import Dict, List, Tuple
import time

# Common valid licenses (SPDX identifiers and common variations)
VALID_LICENSES = {
    'MIT', 'MIT License', 'MIT (Godot demo projects)', 'MIT (code), CC0 assets',
    'MIT (code)', 'MIT License', 'MIT license',
    'CC0', 'CC0-1.0', 'CC0/CC-BY', 'Code CC0; assets CC0/CC-BY',
    'CC-BY', 'CC-BY-4.0', 'CC-BY-SA', 'CC-BY-SA-4.0',
    'GPL', 'GPL-2.0', 'GPL-3.0', 'GPL-2.0+', 'GPL-3.0+',
    'LGPL', 'LGPL-2.1', 'LGPL-3.0',
    'Apache', 'Apache-2.0', 'Apache License 2.0',
    'BSD', 'BSD-2-Clause', 'BSD-3-Clause',
    'ISC',
    'Unlicense',
    'Public Domain',
    'Proprietary',
    'Custom',
    'Unknown'
}

# License patterns that are acceptable (for flexible matching)
LICENSE_PATTERNS = [
    r'MIT',
    r'CC0',
    r'CC-BY',
    r'GPL',
    r'LGPL',
    r'Apache',
    r'BSD',
    r'ISC',
    r'Unlicense',
    r'Public Domain',
    r'Proprietary',
    r'Custom',
    r'Unknown'
]


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
    
    # Check if URL is accessible
    try:
        # Use HEAD request first (more efficient)
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code < 400:
            return True, f"OK ({response.status_code})"
        
        # If HEAD fails, try GET
        response = requests.get(url, timeout=timeout, allow_redirects=True, stream=True)
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


def validate_license(license_str: str) -> Tuple[bool, str]:
    """
    Validate license string.
    Returns (is_valid, warning_message)
    """
    if not license_str or not isinstance(license_str, str):
        return False, "License is empty or not a string"
    
    license_upper = license_str.upper()
    
    # Check for exact match (case-insensitive)
    for valid_license in VALID_LICENSES:
        if license_str == valid_license or license_upper == valid_license.upper():
            return True, ""
    
    # Check for pattern matches (flexible)
    import re
    for pattern in LICENSE_PATTERNS:
        if re.search(pattern, license_str, re.IGNORECASE):
            return True, f"Contains '{pattern}' but not exact match"
    
    # If no match found
    return False, f"License '{license_str}' not recognized. Consider using standard SPDX identifiers."


def validate_entry(entry: Dict, index: int) -> Dict:
    """
    Validate a single entry from base_games.json
    Returns validation results dictionary
    """
    entry_id = entry.get('id', f'entry_{index}')
    title = entry.get('title', 'Unknown')
    
    results = {
        'id': entry_id,
        'title': title,
        'index': index,
        'url_valid': False,
        'url_error': '',
        'license_valid': False,
        'license_warning': '',
        'all_valid': False
    }
    
    # Validate URL
    repo_url = entry.get('repo_or_page', '')
    if repo_url:
        url_valid, url_error = validate_url(repo_url)
        results['url_valid'] = url_valid
        results['url_error'] = url_error
    else:
        results['url_error'] = "Missing 'repo_or_page' field"
    
    # Validate license
    license_str = entry.get('license', '')
    if license_str:
        license_valid, license_warning = validate_license(license_str)
        results['license_valid'] = license_valid
        results['license_warning'] = license_warning
    else:
        results['license_warning'] = "Missing 'license' field"
    
    # Overall validation
    results['all_valid'] = results['url_valid'] and results['license_valid']
    
    return results


def main():
    """Main validation function"""
    json_file = 'base_games.json'
    
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
    invalid_licenses = []
    
    # Validate each entry
    for i, entry in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] Validating: {entry.get('title', 'Unknown')}...", end=' ', flush=True)
        
        result = validate_entry(entry, i)
        results.append(result)
        
        if result['all_valid']:
            print("✅")
            valid_count += 1
        else:
            print("❌")
            if not result['url_valid']:
                invalid_urls.append(result)
            if not result['license_valid']:
                invalid_licenses.append(result)
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total entries: {len(entries)}")
    print(f"✅ Fully valid: {valid_count}")
    print(f"❌ Invalid URLs: {len(invalid_urls)}")
    print(f"⚠️  Invalid licenses: {len(invalid_licenses)}")
    
    # Print invalid URLs
    if invalid_urls:
        print("\n" + "-"*80)
        print("INVALID URLS:")
        print("-"*80)
        for result in invalid_urls:
            print(f"\n  [{result['index']+1}] {result['title']} (ID: {result['id']})")
            print(f"      URL: {entries[result['index']].get('repo_or_page', 'N/A')}")
            print(f"      Error: {result['url_error']}")
    
    # Print invalid licenses
    if invalid_licenses:
        print("\n" + "-"*80)
        print("INVALID OR WARNING LICENSES:")
        print("-"*80)
        for result in invalid_licenses:
            print(f"\n  [{result['index']+1}] {result['title']} (ID: {result['id']})")
            print(f"      License: {entries[result['index']].get('license', 'N/A')}")
            print(f"      Warning: {result['license_warning']}")
    
    # Print all results in detail
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    for result in results:
        status = "✅" if result['all_valid'] else "❌"
        print(f"\n{status} [{result['index']+1}] {result['title']}")
        print(f"   ID: {result['id']}")
        print(f"   URL: {'✅ Valid' if result['url_valid'] else '❌ Invalid'} - {result['url_error']}")
        print(f"   License: {'✅ Valid' if result['license_valid'] else '⚠️  Warning'} - {result['license_warning'] or 'OK'}")
    
    # Exit code
    if valid_count == len(entries):
        print("\n✅ All entries are valid!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(entries) - valid_count} entry/entries have issues")
        sys.exit(1)


if __name__ == '__main__':
    main()

