#!/usr/bin/env python
"""
Launch script for Junon Time Series application.

Usage:
    python run_app.py
    
Or with custom port:
    python run_app.py --port 8502
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Ensure we're in the project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Add project root to PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(project_root) + (os.pathsep + pythonpath if pythonpath else '')
    
    # Parse arguments
    port = "8501"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 2 < len(sys.argv):
            port = sys.argv[i + 2]
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(project_root / "dashboard" / "training" / "Home.py"),
        "--server.port", port,
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"Starting Junon Time Series on http://localhost:{port}")
    print(f"Project root: {project_root}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
