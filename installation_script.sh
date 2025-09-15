#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# --- Install Node.js ---
echo "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs build-essential

# Verify Node.js installation
node -v
npm -v

# --- Install Python 3 and venv ---
echo "Installing Python3, pip, and venv..."
sudo apt install -y python3 python3-pip python3-venv

# Verify Python installation
python3 --version
pip3 --version

# --- Frontend setup ---
if [ -d "frontend" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
else
    echo "Frontend directory not found, skipping..."
fi

# --- Backend setup ---
if [ -d "backend" ]; then
    echo "Setting up Python virtual environment for backend..."
    cd backend

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip inside venv
    pip install --upgrade pip

    # Install backend requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "requirements.txt not found in backend, skipping..."
    fi

    deactivate
    cd ..
else
    echo "Backend directory not found, skipping..."
fi

echo "Setup complete!"
