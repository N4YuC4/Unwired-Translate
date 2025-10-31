import json
import os
from datetime import datetime

HISTORY_FILE = "artifacts/translation_history.json"

def _get_history_file_path():
    """Returns the absolute path to the history file."""
    # Assuming the script is run from the project root or app/main.py
    # Adjust path if necessary based on where this utility is called from
    return os.path.abspath(HISTORY_FILE)

def load_history():
    """
    Loads translation history from a JSON file.
    Returns an empty list if the file does not exist or is invalid.
    """
    file_path = _get_history_file_path()
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            history = json.load(f)
            # Ensure all entries have necessary keys, or filter them out
            history = [entry for entry in history if all(k in entry for k in ["english", "turkish", "timestamp"])]
            return history
    except json.JSONDecodeError:
        print(f"Warning: History file '{file_path}' is corrupted. Starting with empty history.")
        return []
    except Exception as e:
        print(f"Error loading history from '{file_path}': {e}")
        return []

def save_history(history):
    """
    Saves translation history to a JSON file.
    """
    file_path = _get_history_file_path()
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history to '{file_path}': {e}")

def add_to_history(english_text, turkish_text):
    """
    Adds a new translation entry to the history and saves it.
    """
    history = load_history()
    new_entry = {
        "english": english_text,
        "turkish": turkish_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    history.insert(0, new_entry) # Add to the beginning for easier display of latest translations
    # Keep only the last N entries to prevent the file from growing too large
    max_history_entries = 100 # Configurable limit
    if len(history) > max_history_entries:
        history = history[:max_history_entries]
    save_history(history)

# Ensure the artifacts directory exists for the history file
os.makedirs(os.path.dirname(_get_history_file_path()), exist_ok=True)
