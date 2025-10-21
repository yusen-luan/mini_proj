from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
from uuid import uuid4


def to_uppercase_filename(name: str) -> str:
    """
    Uppercase only alphabetic characters in the base filename, preserving the
    original extension case. Numbers and non-letter symbols are left unchanged.
    """
    p = Path(name)
    suffix = "".join(p.suffixes)
    base = name[: -len(suffix)] if suffix else name
    transformed_base = "".join(ch.upper() if ch.isalpha() else ch for ch in base)
    return transformed_base + suffix


def safe_case_change_rename(source: Path, target: Path) -> None:
    """
    Perform a case-only rename safely on case-insensitive filesystems (e.g., Windows)
    by renaming through a temporary unique filename first.
    """
    temp_name = f".__tmp_case__{uuid4().hex}__{source.name}"
    temp_path = source.with_name(temp_name)
    source.rename(temp_path)
    temp_path.rename(target)


def rename_files_to_uppercase(folder: Path) -> Tuple[int, int]:
    """
    Rename all regular files in 'folder' so that only letters in their names are uppercased.
    Returns (renamed_count, skipped_count).
    """
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    renamed_count = 0
    skipped_count = 0

    for entry in folder.iterdir():
        if not entry.is_file():
            continue

        new_name = to_uppercase_filename(entry.name)
        if new_name == entry.name:
            skipped_count += 1
            continue

        target = entry.with_name(new_name)

        # Case-only change handling on case-insensitive filesystems
        if entry.name.lower() == new_name.lower():
            safe_case_change_rename(entry, target)
            renamed_count += 1
            continue

        if target.exists():
            # Avoid collisions
            skipped_count += 1
            continue

        entry.rename(target)
        renamed_count += 1

    return renamed_count, skipped_count


def lowercase_extension_in_name(name: str) -> str:
    """
    Return a filename where the part after the last '.' is lowercased.
    If no '.' exists, return the name unchanged.
    """
    dot_index = name.rfind(".")
    if dot_index == -1:
        return name
    base = name[:dot_index]
    ext = name[dot_index + 1 :]
    return f"{base}.{ext.lower()}"


def lowercase_extensions(folder: Path) -> Tuple[int, int]:
    """
    Lowercase the extension (text after the last '.') for all regular files in folder.
    Returns (renamed_count, skipped_count).
    """
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    renamed_count = 0
    skipped_count = 0

    for entry in folder.iterdir():
        if not entry.is_file():
            continue

        new_name = lowercase_extension_in_name(entry.name)
        if new_name == entry.name:
            skipped_count += 1
            continue

        target = entry.with_name(new_name)

        # Case-only change handling on case-insensitive filesystems
        if entry.name.lower() == new_name.lower():
            safe_case_change_rename(entry, target)
            renamed_count += 1
            continue

        if target.exists():
            # Avoid collisions
            skipped_count += 1
            continue

        entry.rename(target)
        renamed_count += 1

    return renamed_count, skipped_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert filenames in a folder to uppercase (letters only), "
            "ignoring numbers and other symbols."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Target folder containing files to rename (default: current directory)",
    )
    parser.add_argument(
        "--lower-ext-only",
        action="store_true",
        help=(
            "Lowercase the extension (after the last '.') for files in the folder "
            "without changing the base filename."
        ),
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if args.lower_ext_only:
        renamed, skipped = lowercase_extensions(folder)
    else:
        renamed, skipped = rename_files_to_uppercase(folder)
    print(f"Renamed: {renamed}, Skipped: {skipped} in '{folder}'.")


if __name__ == "__main__":
    main()


