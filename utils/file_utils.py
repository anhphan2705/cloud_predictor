import glob

def get_file_paths(dir: str) -> list:
    """
    Retrieves a list of files matching the specified directory pattern.
    
    Parameters:
    dir (str): The directory pattern to search for files.

    Usage:
    file_paths = get_file_paths('data/*.nc')

    Returns:
    list: A list of file paths that match the specified directory pattern.
    """
    files = glob.glob(dir)
    print(f"[INFO] Found {len(files)} files in {dir}")
    return files