@echo off
REM Script to run both main.py and predictor_app.py in parallel on Windows
REM Usage: start_parallel.bat

echo 🚀 Starting Aviator System in Parallel Mode...
echo ==============================================
echo.

REM Start main.py in new window
echo ▶️  Starting data collection (main.py)...
start "Data Collection" python main.py

REM Wait a moment
timeout /t 3 /nobreak >nul

REM Start predictor_app.py in new window
echo ▶️  Starting predictor app (predictor_app.py)...
start "Predictor App" python predictor_app.py

echo.
echo ==============================================
echo ✅ Both processes started in separate windows!
echo 📊 Data Collection: Running in window 1
echo 🤖 Predictor App: Running in window 2
echo 🌐 Web Interface: http://localhost:5001
echo ==============================================
echo.
echo ⚠️  Close the windows to stop the processes
echo.

pause
