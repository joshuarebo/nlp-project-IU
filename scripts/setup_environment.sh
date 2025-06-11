#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install packages from requirements.txt
echo "Installing required packages..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "Environment setup complete! You can now run the notebook."
echo "To activate this environment in the future, run: source venv/bin/activate"
