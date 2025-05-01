import re
from pathlib import Path

import readtime
from slugify import slugify

from brainiac.model import MetadataFile


def read_file(file_path: Path) -> str:
    """
    Read and return the content of a json file.

    Args:
        file_path (str): Path to the file to be read

    Returns:
        str: Content of the file

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be accessed due to permissions
    """
    try:
        with open(file_path, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except PermissionError:
        raise PermissionError(f"Permission denied when trying to read {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while reading {file_path}: {str(e)}")


def write_file(file_path: Path, content: MetadataFile):
    """
    Write the content to a json file. Always overwrites the file if it exists.

    Args:
        file_path (str): Path to the file to be written
        content (str): Content to be written to the file

    Raises:
        PermissionError: If the file cannot be accessed due to permissions
    """
    try:
        with open(file_path, "w") as file:
            file.write(content.model_dump_json(indent=2))

    except PermissionError:
        raise PermissionError(f"Permission denied when trying to write {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while writing to {file_path}: {str(e)}")


def copy_file(dest: Path, content: str):
    """
    Copy the content to a file.

    Args:
        dest (str): Path to the file to be written
        content (str): Content to be written to the file

    Raises:
        PermissionError: If the file cannot be accessed due to permissions
    """
    try:
        with open(dest, "x") as file:
            file.write(content)

    except PermissionError:
        raise PermissionError(f"Permission denied when trying to write {dest}")
    except Exception as e:
        raise Exception(f"An error occurred while writing to {dest}: {str(e)}")


def convert_to_slug(title: str) -> str:
    return slugify(title)


def get_reading_time_in_minutes(content: str) -> int:
    return readtime.of_markdown(content).minutes


def get_word_count(content: str) -> int:
    # Counting words using regex
    words = re.findall(r"\b\w+\b", content)
    return len(words)
