#!/usr/bin/env python3
"""
Script to run main.py (data collection) and predictor_app.py (prediction) in parallel.
"""

import subprocess
import sys
import time
import signal
import os

processes = []

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n⚠️  Shutting down all processes...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    print("🚀 Starting parallel execution...")
    print("=" * 60)
    print("📊 Process 1: Data Collection (main.py)")
    print("🤖 Process 2: Predictor App (predictor_app.py)")
    print("=" * 60)
    
    # Start main.py (data collection) - with visible output
    print("\n▶️  Starting data collection...")
    proc1 = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    processes.append(proc1)
    print("✅ Data collection started (PID: {})".format(proc1.pid))
    
    # Wait a bit for main.py to initialize
    time.sleep(3)
    
    # Start predictor_app.py (prediction server) - with visible output
    print("\n▶️  Starting predictor app...")
    proc2 = subprocess.Popen(
        [sys.executable, "predictor_app.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    processes.append(proc2)
    print("✅ Predictor app started (PID: {})".format(proc2.pid))
    
    print("\n" + "=" * 60)
    print("✅ Both processes running in parallel!")
    print("📊 Data Collection: Running in background")
    print("🌐 Predictor App: http://localhost:5001")
    print("=" * 60)
    print("\n⚠️  Press Ctrl+C to stop all processes\n")
    
    # Monitor processes
    try:
        while True:
            # Check if processes are still alive
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    print(f"⚠️  Process {i+1} exited with code {proc.returncode}")
                    if proc.returncode != 0:
                        stdout, stderr = proc.communicate()
                        if stderr:
                            print(f"Error: {stderr}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()
