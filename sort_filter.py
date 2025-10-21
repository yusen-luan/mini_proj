from pathlib import Path
import shutil
from typing import List


def list_sorted_files(directory: Path) -> List[Path]:
    """
    Return a list of files in the given directory, sorted lexicographically by filename.
    Only regular files are included.
    """
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist or is not a directory: {directory}")

    files = [p for p in directory.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def copy_files(files: List[Path], destination_dir: Path) -> None:
    """
    Copy the provided files into destination_dir, preserving filenames.
    Creates destination_dir if it does not exist.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_path in files:
        target_path = destination_dir / source_path.name
        shutil.copy2(source_path, target_path)


def main() -> None:
    base_dir = Path(__file__).parent
    train_dir = base_dir / "train"
    output_dir = base_dir / "relabelled_yusen"

    all_files = list_sorted_files(train_dir)

    # Select files from index 6401 to the end (0-based index)
    start_index = 6401
    selected_files = all_files[start_index:]

    if not selected_files:
        print("No files selected. Nothing to copy.")
        return

    copy_files(selected_files, output_dir)
    print(f"Copied {len(selected_files)} files to '{output_dir}'.")


if __name__ == "__main__":
    main()


