#!/bin/bash

# Script to run both main.py and predictor_app.py in parallel
# Usage: ./start_parallel.sh

echo "🚀 Starting Aviator System in Parallel Mode..."
echo "=============================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "⚠️  Shutting down all processes..."
    kill $PID1 $PID2 2>/dev/null
    wait $PID1 $PID2 2>/dev/null
    echo "✅ All processes stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Start main.py in background
echo "▶️  Starting data collection (main.py)..."
python3 main.py &
PID1=$!
echo "✅ Data collection started (PID: $PID1)"
echo ""

# Wait a moment for initialization
sleep 3

# Start predictor_app.py in background
echo "▶️  Starting predictor app (predictor_app.py)..."
python3 predictor_app.py &
PID2=$!
echo "✅ Predictor app started (PID: $PID2)"
echo ""

echo "=============================================="
echo "✅ Both processes running in parallel!"
echo "📊 Data Collection: PID $PID1"
echo "🤖 Predictor App: PID $PID2"
echo "🌐 Web Interface: http://localhost:5001"
echo "=============================================="
echo ""
echo "⚠️  Press Ctrl+C to stop all processes"
echo ""

# Wait for both processes
wait $PID1 $PID2
