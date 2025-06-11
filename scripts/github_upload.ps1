# Create necessary directories
New-Item -Path "data", "notebooks", "scripts", "results/visualizations", "results/models" -ItemType Directory -Force | Out-Null

# Copy notebook to notebooks directory
Copy-Item -Path "nlp_topic_modeling_pipeline.ipynb" -Destination "notebooks/" -Force

# Create .gitignore file
@"
# Ignore large data files and outputs
*.csv
*.pkl
*.model
__pycache__/
.ipynb_checkpoints/
venv/
"@ | Set-Content -Path ".gitignore"

# Initialize git repository if not already initialized
if (-not (Test-Path ".git")) {
    git init
}

# Add files to git
git add README.md
git add requirements.txt
git add notebooks/
git add scripts/
git add .gitignore

# Try to add visualization files if they exist
git add results/visualizations/*.png 2>$null
git add results/visualizations/*.html 2>$null
git add results/results_summary.json 2>$null

# Commit changes
git commit -m "Initial commit of NLP topic modeling pipeline"

# Add remote repository
git remote add origin https://github.com/joshuarebo/nlp-project-IU.git

# Push to GitHub
Write-Host "Ready to push to GitHub. Run the following command:"
Write-Host "git push -u origin main"
