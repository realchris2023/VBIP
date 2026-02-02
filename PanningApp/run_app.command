#!/bin/bash
cd "$(dirname "$0")"

echo "--- AUTO-SETUP STARTED ---"

# 1. Check for Python command
PY_CMD="python3"
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found in PATH. Checking for 'python'..."
    if command -v python &> /dev/null; then
        PY_CMD="python"
    else
        echo "ERROR: Python is not installed on this Mac."
        echo "Please download it from https://www.python.org/downloads/"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# 2. Create/Repair Virtual Environment
if [ ! -f "venv/bin/python3" ]; then
    echo "Creating new virtual environment using $PY_CMD..."
    rm -rf venv  # Remove any broken leftovers
    $PY_CMD -m venv venv
    
    if [ ! -f "venv/bin/python3" ]; then
        echo "CRITICAL ERROR: Failed to create venv."
        read -p "Press Enter to exit..."
        exit 1
    fi

    echo "Installing requirements..."
    ./venv/bin/pip install -r requirements.txt
fi

# 3. Run the App
echo "Launching App..."
./venv/bin/python3 src/main.py

# 4. Catch Errors
if [ $? -ne 0 ]; then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " The app crashed. Read the error message above."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    read -p "Press Enter to close..."
fi