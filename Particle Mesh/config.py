"""
Single place that loads config.json (resolved next to this file, so the working directory doesn't matter).
Every module imports CONFIG from here instead of opening the file itself.
"""
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)
