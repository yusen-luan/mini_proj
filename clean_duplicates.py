import re
from pathlib import Path
from collections import defaultdict


def extract_base_name(filename):
    """
    Extract the base name before the -0 suffix.
    
    Args:
        filename: Name of the file (e.g., "ABC123-0.png" or "ABC123-0 (1).png")
        
    Returns:
        str: Base name (e.g., "ABC123") or None if pattern doesn't match
    """
    # Match pattern: <base_name>-0.png or <base_name>-0 (x).png
    # Pattern: anything followed by -0, then optionally space and (number), then .png
    pattern = re.compile(r'^(.+?)-0(?:\s\(\d+\))?\.png$')
    match = pattern.match(filename)
    
    if match:
        return match.group(1)  # Return the base name part
    return None


def find_duplicates(directory):
    """
    Find duplicate files based on the base name before -0 suffix.
    
    Args:
        directory: Path to directory to scan
        
    Returns:
        dict: Dictionary mapping base names to list of duplicate files
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist!")
        return {}
    
    # Find all PNG files
    all_png_files = list(directory.glob("*.png"))
    
    # Group files by their base name
    base_name_groups = defaultdict(list)
    files_without_pattern = []
    
    for file_path in all_png_files:
        base_name = extract_base_name(file_path.name)
        if base_name:
            base_name_groups[base_name].append(file_path.name)
        else:
            files_without_pattern.append(file_path.name)
    
    # Filter to only groups with duplicates (more than 1 file)
    duplicates = {base: files for base, files in base_name_groups.items() if len(files) > 1}
    
    return duplicates, files_without_pattern, len(all_png_files)


def main():
    """Main function to find duplicate files."""
    # Directory to scan
    data_dir = "./data/relabelled"
    
    print("="*80)
    print("DUPLICATE FILE FINDER")
    print("="*80)
    print("This script finds files with the same base name (before -0 suffix)")
    print(f"Target directory: {data_dir}")
    print("="*80)
    
    print(f"\nScanning directory: {data_dir}")
    
    # Find duplicates
    duplicates, files_without_pattern, total_files = find_duplicates(data_dir)
    
    print(f"Total PNG files found: {total_files}")
    print(f"Files without -0 pattern: {len(files_without_pattern)}")
    print(f"Base names with duplicates: {len(duplicates)}")
    
    if duplicates:
        print("\n" + "="*80)
        print("DUPLICATE GROUPS FOUND")
        print("="*80)
        
        total_duplicate_files = 0
        for base_name, files in sorted(duplicates.items()):
            print(f"\nBase name: '{base_name}' ({len(files)} files)")
            print("-" * 80)
            for filename in sorted(files):
                print(f"  - {filename}")
            total_duplicate_files += len(files)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Number of base names with duplicates: {len(duplicates)}")
        print(f"Total files involved in duplicates: {total_duplicate_files}")
        print(f"Extra duplicate files (could be removed): {total_duplicate_files - len(duplicates)}")
        print("="*80)
        
        # Show which specific files could be deleted (the ones with (x) suffix)
        print("\n" + "="*80)
        print("FILES THAT COULD BE SAFELY DELETED (ones with (x) suffix)")
        print("="*80)
        
        deletable_pattern = re.compile(r'^.*-0\s\(\d+\)\.png$')
        deletable_files = []
        
        for base_name, files in sorted(duplicates.items()):
            for filename in sorted(files):
                if deletable_pattern.match(filename):
                    deletable_files.append(filename)
                    print(f"  - {filename}")
        
        print(f"\nTotal files that could be deleted: {len(deletable_files)}")
        print("="*80)
        
    else:
        print("\nNo duplicate files found!")
    
    if files_without_pattern:
        print(f"\nWarning: {len(files_without_pattern)} files don't follow the expected '-0.png' naming pattern:")
        for filename in sorted(files_without_pattern)[:10]:  # Show first 10
            print(f"  - {filename}")
        if len(files_without_pattern) > 10:
            print(f"  ... and {len(files_without_pattern) - 10} more")


if __name__ == "__main__":
    main()

