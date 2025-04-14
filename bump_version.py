#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to bump the version number in flowtracks/__init__.py
Usage: python bump_version.py [major|minor|patch]
"""

import re
import sys
from pathlib import Path

def bump_version(version_type):
    """
    Bump the version number in __init__.py
    
    Args:
        version_type: One of 'major', 'minor', or 'patch'
    """
    init_file = Path('flowtracks/__init__.py')
    content = init_file.read_text()
    
    # Find the current version
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]", content)
    if not version_match:
        print("Error: Could not find version in flowtracks/__init__.py")
        sys.exit(1)
    
    current_version = version_match.group(1)
    print(f"Current version: {current_version}")
    
    # Split the version into components
    try:
        major, minor, patch = map(int, current_version.split('.'))
    except ValueError:
        print(f"Error: Version {current_version} does not follow semantic versioning (major.minor.patch)")
        sys.exit(1)
    
    # Bump the appropriate component
    if version_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif version_type == 'minor':
        minor += 1
        patch = 0
    elif version_type == 'patch':
        patch += 1
    else:
        print(f"Error: Invalid version type '{version_type}'. Use 'major', 'minor', or 'patch'")
        sys.exit(1)
    
    # Create the new version string
    new_version = f"{major}.{minor}.{patch}"
    print(f"New version: {new_version}")
    
    # Replace the version in the file
    new_content = re.sub(
        r"__version__\s*=\s*['\"]([^'\"]*)['\"]",
        f"__version__ = '{new_version}'",
        content
    )
    
    # Write the updated content back to the file
    init_file.write_text(new_content)
    print(f"Version bumped to {new_version}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_version(sys.argv[1])
