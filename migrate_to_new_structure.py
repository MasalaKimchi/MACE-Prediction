#!/usr/bin/env python3
"""
Migration script to help transition from old structure to new compartmentalized structure.
This script will update import statements in existing scripts to use the new structure.
"""

import os
import re
from pathlib import Path


def update_imports_in_file(file_path: str) -> bool:
    """Update import statements in a file to use the new structure."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Update imports
        replacements = [
            (r'from dataset import', 'from dataloaders import'),
            (r'from network import', 'from architectures import'),
            (r'from preprocessing import', 'from data import'),
            (r'import dataset', 'import dataloaders'),
            (r'import network', 'import architectures'),
            (r'import preprocessing', 'import data'),
        ]
        
        for old_pattern, new_pattern in replacements:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Updated imports in: {file_path}")
            return True
        else:
            print(f"No changes needed in: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def main():
    """Main migration function."""
    print("Survival Analysis Project - Structure Migration")
    print("=" * 50)
    
    # Files to update
    files_to_update = [
        'test_dataset.py',
        'test_monai_survival_dataset.py', 
        'test_network.py',
        'test_preprocessing.py'
    ]
    
    updated_count = 0
    for file_path in files_to_update:
        if os.path.exists(file_path):
            if update_imports_in_file(file_path):
                updated_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nMigration complete! Updated {updated_count} files.")
    print("\nNext steps:")
    print("1. Test your updated scripts")
    print("2. Consider using the new config-based training approach")
    print("3. Remove old files when you're confident everything works")
    print("\nNew structure benefits:")
    print("- Clean separation of concerns")
    print("- Easier to maintain and extend")
    print("- Config-based experiments for reproducibility")
    print("- Modular design following SegFormer3D pattern")


if __name__ == '__main__':
    main()
