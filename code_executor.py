import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def detect_dependencies(code):
    """
    Detect common ML libraries used in the code.

    Args:
        code: Python code string

    Returns:
        List of detected dependencies
    """
    dependencies = []
    common_imports = {
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'plotly': 'plotly',
        'keras': 'keras',
        'scipy': 'scipy',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm'
    }

    for import_name, package_name in common_imports.items():
        if f"import {import_name}" in code or f"from {import_name}" in code:
            dependencies.append(package_name)

    return list(set(dependencies))  # Remove duplicates


def save_code_to_file(code, topic):
    """
    Save generated code to a file

    Args:
        code: Python code string
        topic: Topic name for filename

    Returns:
        Filename of saved code or None if failed
    """
    if not code or not code.strip():
        logger.warning("Cannot save empty code.")
        return None
