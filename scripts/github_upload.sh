#!/bin/bash

# Make sure all necessary directories exist
if [ ! -d "data" ]; then
    mkdir -p data
fi

if [ ! -d "notebooks" ]; then
    mkdir -p notebooks
fi

if [ ! -d "scripts" ]; then
    mkdir -p scripts
fi

if [ ! -d "results/visualizations" ]; then
    mkdir -p results/visualizations
fi

if [ ! -d "results/models" ]; then
    mkdir -p results/models
fi

# Copy notebook to notebooks directory
cp nlp_topic_modeling_pipeline.ipynb notebooks/

# Create .gitignore file
echo "# Ignore large data files and outputs" > .gitignore
echo "*.csv" >> .gitignore
echo "*.pkl" >> .gitignore
echo "*.model" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
echo "venv/" >> .gitignore

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    git init
fi

# Add files to git
git add README.md
git add requirements.txt
git add notebooks/
git add scripts/
git add .gitignore
git add results/visualizations/*.png || true
git add results/visualizations/*.html || true
git add results/results_summary.json || true

# Commit changes
git commit -m "Initial commit of NLP topic modeling pipeline"

# Add remote repository
git remote add origin https://github.com/joshuarebo/nlp-project-IU.git

# Push to GitHub
echo "Ready to push to GitHub. Run the following command:"
echo "git push -u origin main"
