#!/bin/bash
# Windows-Specific Deploy Script

echo "--- DEPLOYMENT DIAGNOSTIC STARTING ---"

# 1. FIND A WORKING PYTHON
# We test 'py', 'python', and 'python3' to see which one actually responds.
# The 'tail' check filters out the Microsoft Store shim which hangs or errors.

if py --version > /dev/null 2>&1; then
    PY_CMD="py"
elif python --version > /dev/null 2>&1; then
    PY_CMD="python"
elif python3 --version > /dev/null 2>&1; then
    PY_CMD="python3"
else
    echo "CRITICAL ERROR: No working Python found!"
    echo "Windows is blocking 'python'. Please do this:"
    echo "1. Press Start key -> Type 'App Execution Aliases'"
    echo "2. Turn OFF the toggles for 'App Installer (python.exe)' and 'python3.exe'"
    exit 1
fi

echo "FOUND WORKING PYTHON: $PY_CMD"

# 2. CHECK FOR PIP
if ! $PY_CMD -m pip --version > /dev/null 2>&1; then
    echo "Pip missing. Installing via ensurepip..."
    $PY_CMD -m ensurepip --default-pip
fi

# 3. INSTALL DEPENDENCIES (Global/User mode to avoid venv issues)
echo "Installing dependencies..."
# We skip venv entirely to stop the cycle of errors. We just install to your user profile.
$PY_CMD -m pip install --user --upgrade pip
$PY_CMD -m pip install --user -r requirements.txt

# 4. KILL OLD PROCESSES (Prevent "Address already in use")
# This finds any process using port 8080 and kills it (Windows specific)
if command -v netstat > /dev/null; then
     PID=$(netstat -ano | grep :8080 | awk '{print $5}' | head -n 1)
     if [ -n "$PID" ] && [ "$PID" != "0" ]; then
         echo "Killing old server on process $PID..."
         taskkill //F //PID $PID > /dev/null 2>&1 || true
     fi
fi

# 5. START SERVER
echo "Starting Server..."
# We use standard python execution (no nohup needed for basic test)
$PY_CMD main.py

# Note: The script will stay open showing logs. Press Ctrl+C to stop.