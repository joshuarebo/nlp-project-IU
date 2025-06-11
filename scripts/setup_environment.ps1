# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

# Install packages from requirements.txt
Write-Host "Installing required packages..." -ForegroundColor Green
pip install -r requirements.txt

# Download spaCy model
Write-Host "Downloading spaCy model..." -ForegroundColor Green
python -m spacy download en_core_web_sm

Write-Host "Environment setup complete! You can now run the notebook." -ForegroundColor Green
Write-Host "To activate this environment in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
