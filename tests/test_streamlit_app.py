"""
Streamlit AppTest Suite for Crop_AI (Stage 10)
Tests the Streamlit app components using Streamlit's built-in AppTest framework.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit.testing.v1 import AppTest

def test_app_initial_state():
    at = AppTest.from_file(str(ROOT_DIR / "app.py"))
    at.run(timeout=30)
    
    # Check title and subheader
    assert not at.exception, f"App encountered exception: {at.exception}"
    assert at.title[0].value == "Crop_AI"
    assert at.subheader[0].value == "Soybean Disease & Health Detection"
    
    # Check initial info message when no image is uploaded
    assert any("Please upload a soybean image." in info.value for info in at.info)
    print("[OK] Initial app state verified: Title, Subtitle, File uploader, and info prompt present.")

if __name__ == "__main__":
    test_app_initial_state()
    print("Streamlit AppTest passed!")
