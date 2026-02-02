#!/usr/bin/env python
"""
Launch script for Junon Time Series application.

Usage:
    python run_app.py              # Streamlit + MLflow UI (même base mlflow.db)
    python run_app.py --no-mlflow  # Streamlit uniquement
    python run_app.py --port 8502  # Port personnalisé
"""

import subprocess
import sys
import os
import argparse
import time
import socket
from pathlib import Path


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if TCP port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def main():
    parser = argparse.ArgumentParser(
        description="Launch Junon Time Series Streamlit application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py
  python run_app.py --mlflow
  python run_app.py --port 8502
  python run_app.py --port 8501 --host 0.0.0.0
        """
    )
    parser.add_argument("--port", type=int, default=8501,
                       help="Port to run Streamlit on (default: 8501)")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Host to bind to (default: localhost)")
    parser.add_argument("--mlflow", action="store_true",
                       help="Launch MLflow UI alongside Streamlit (default: True)")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Do not launch MLflow UI")
    
    args = parser.parse_args()
    launch_mlflow = not args.no_mlflow
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Add project root to PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(project_root) + (os.pathsep + pythonpath if pythonpath else '')
    
    # MLflow : même base pour l'app et l'UI
    mlflow_db = project_root / "mlflow.db"
    env["MLFLOW_TRACKING_URI"] = f"sqlite:///{mlflow_db.as_posix()}"
    # Forcer un seul worker Uvicorn (Windows) pour éviter WinError 10022
    env["MLFLOW_SERVER_WORKERS"] = "1"
    
    # Launch Streamlit
    app_path = project_root / "dashboard" / "training" / "Home.py"
    
    if not app_path.exists():
        print(f"Error: Application file not found at {app_path}")
        sys.exit(1)
    
    mlflow_process = None
    if launch_mlflow:
        if _is_port_in_use(5000):
            print("MLflow UI already running on port 5000; skipping launch.")
            launch_mlflow = False
    if launch_mlflow:
        print("Starting MLflow UI...")
        try:
            mlflow_cmd = [
                sys.executable, "-m", "mlflow", "ui", "--port", "5000",
                "--host", "127.0.0.1",
                "--workers", "1",
                "--backend-store-uri", env["MLFLOW_TRACKING_URI"],
            ]
            mlflow_process = subprocess.Popen(mlflow_cmd, env=env)
            # Give it a moment to verify it started
            time.sleep(2)
            if mlflow_process.poll() is None:
                print("MLflow UI started at http://localhost:5000")
            else:
                print("Warning: MLflow UI failed to start immediately.")
        except Exception as e:
            print(f"Failed to start MLflow: {e}")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--browser.gatherUsageStats", "false"
    ]
    
    print("=" * 60)
    print("Junon Time Series - Starting Application")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Application: {app_path}")
    print(f"Streamlit URL: http://{args.host}:{args.port}")
    if launch_mlflow:
        print(f"MLflow UI URL: http://localhost:5000")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting app: {e}")
        # sys.exit(1) # Cleanup first
    finally:
        if mlflow_process:
            print("Stopping MLflow UI...")
            mlflow_process.terminate()
            try:
                mlflow_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mlflow_process.kill()

if __name__ == "__main__":
    main()
