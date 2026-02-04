#!/bin/bash

# Quick start script for Jarvis

echo "================================"
echo "Jarvis Voice Assistant - Quick Start"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Check if Piper is installed
if [ ! -f "piper/piper/piper" ]; then
    echo "Piper not found. Running setup..."
    python setup_piper.py
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from example..."
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env to configure your settings!"
    echo ""
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "⚠️  Ollama doesn't appear to be running!"
    echo "   Start it with: ollama serve"
    echo "   Or install from: https://ollama.ai/download"
    echo ""
fi

# Start Jarvis
echo ""
echo "Starting Jarvis..."
echo "Modes: --tui (Terminal UI), --text (text-only)"
echo ""
python main.py "$@"
