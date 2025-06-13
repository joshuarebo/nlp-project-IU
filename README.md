# NLP Topic Modeling Pipeline

A production-grade, end-to-end topic modeling pipeline for large-scale unstructured text, built for the U.S. Consumer Financial Protection Bureau complaint narratives. This project leverages advanced NLP, classical, and neural topic modeling techniques, with a focus on reproducibility, scalability, and actionable insights.

## Features
- **Robust Preprocessing**: Advanced cleaning, lemmatization, domain-specific stopwords, and quality control.
- **Exploratory Analysis**: Token, n-gram, and word cloud visualizations for rapid data understanding.
- **Classical Topic Modeling**: Optimized LDA and NMF with grid search, coherence evaluation, and interpretability tools.
- **Neural Topic Modeling**: State-of-the-art encoder-decoder architecture using Sentence-BERT, Word2Vec, and custom loss functions for coherence and diversity.
- **Automated Model Selection**: Performance-driven selection and export of best models for deployment.
- **Production Ready**: All artifacts (models, configs, visualizations) saved for seamless deployment.

## Quickstart
1. **Environment Setup**
   - Windows: `./scripts/setup_environment.ps1`
   - Linux/macOS: `bash scripts/setup_environment.sh`
   - Or manually: `python -m venv venv && pip install -r requirements.txt`
   - Or follow instructions in 'INSTRUCTOR_NOTE.md'
2. **Data**: Place `consumer_complaints.csv` in the project root after downloading from https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints/data.
3. **Run**: Launch the notebook `nlp_topic_modeling_pipeline.ipynb` and follow the stepwise workflow.

## Pipeline Overview
1. Environment Setup & Dependency Management
2. Data Ingestion & Validation
3. Text Cleaning & Preprocessing
4. Exploratory Data Analysis
5. Feature Engineering (TF-IDF, Word2Vec)
6. Topic Modeling (LDA, NMF, Neural)
7. Model Evaluation & Visualization
8. Export, Comparison, and Deployment

## Outputs
- **results/enhanced_nmf_analysis.json**: Traditional NMF results
- **results/neural_topic_modeling_results.json**: Neural model results
- **results/models/best_neural_topic_model.pth**: Best performing model
- **results/deployment_config.json**: Deployment configuration
- **results/visualizations/**: All visualizations (topics, word clouds, distributions)

## Technical Highlights
- Enhanced preprocessing and domain adaptation
- Grid search and coherence-based model selection
- Neural topic modeling with multi-embedding support
- Automated export for production deployment
- Scalable to 50k+ documents

## Recommended Usage
- Use Sentence-BERT embeddings for best semantic performance
- Monitor reconstruction loss and topic drift in production
- Periodically retrain with new data for optimal results

---

**Built for scalable, explainable, and production-ready NLP topic modeling.**
**Joshua Rebo**