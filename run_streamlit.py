#!/usr/bin/env python3
"""
Simple script to run the FactShield Streamlit app.

This script provides an easy way to launch the Streamlit application
with proper configuration and error handling.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    streamlit_app_path = script_dir / "streamlit_app.py"
    
    # Check if the streamlit app exists
    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at {streamlit_app_path}")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        subprocess.run(["streamlit", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Streamlit is not installed or not in PATH")
        print("Please install it with: pip install streamlit")
        sys.exit(1)
    
    print("🛡️ Starting FactShield - AI Fact-Checking System...")
    print(f"📁 App location: {streamlit_app_path}")
    print("🌐 The app will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Launch streamlit
    try:
        cmd = [
            "streamlit", "run", str(streamlit_app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ]
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down FactShield...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()