# 🔍 NLP-Based Topic Modeling Pipeline
**Consumer Complaints Analysis using Advanced Topic Modeling**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 👨‍🎓 Student Information
- **Name**: Joshua Rebo
- **Email**: joshua.rebo@iu-study.org  
- **Matriculation Number**: 9213334
- **University**: IU International University of Applied Sciences

## 📋 Project Overview

This repository contains a comprehensive implementation of an **enhanced NLP-based topic modeling pipeline** applied to consumer complaint data from the U.S. Consumer Financial Protection Bureau. The project demonstrates advanced text mining techniques to extract meaningful topics and insights from unstructured consumer complaints.

### 🎯 Objectives
- Extract dominant themes from consumer complaints using state-of-the-art NLP methods
- Compare effectiveness of different topic modeling approaches (LDA vs NMF)
- Implement parameter optimization for better topic coherence
- Provide actionable insights for improving financial services

## 📊 Dataset Overview

- **Source**: U.S. Consumer Financial Protection Bureau Consumer Complaints Database
- **Total Complaints**: 555,957 complaints
- **Processed Dataset**: 49,608 complaints (after filtering and preprocessing)
- **Average Tokens per Complaint**: 82.6 tokens
- **Text Processing**: Comprehensive cleaning, tokenization, lemmatization, and stopword removal

## 🔬 Methodology & Analysis Pipeline

### 1. **Enhanced Text Preprocessing**
- Lowercase normalization and punctuation removal
- Domain-specific stopword filtering  
- Advanced tokenization using spaCy
- Lemmatization for word normalization
- Quality filtering (minimum token thresholds)

### 2. **Advanced Vectorization Techniques**
- **TF-IDF Vectorization**: 1,000 optimized features with n-gram analysis
- **Word2Vec Embeddings**: 100-dimensional dense vectors with 10,626 vocabulary terms
- **Semantic Analysis**: Capturing both statistical and semantic word relationships

### 3. **Enhanced Topic Modeling**

#### **LDA (Latent Dirichlet Allocation)**
- **Topics Extracted**: 8 optimal topics
- **Coherence Score**: 0.4395 (c_v metric)
- **Method**: Probabilistic topic modeling with automatic parameter tuning

#### **Enhanced NMF (Non-negative Matrix Factorization)**
- **Topics Extracted**: 10 optimal topics  
- **Coherence Score**: 0.5837 (**33% better than LDA**)
- **Optimization**: Grid search across multiple parameters
- **Best Parameters**: n_components=10, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.5

## 🏆 Key Findings & Insights

### **Topic Model Performance Comparison**
| Model | Topics | Coherence Score | Reconstruction Error | Best Performance |
|-------|--------|-----------------|---------------------|------------------|
| LDA   | 8      | 0.4395         | N/A                 | Baseline        |
| **Enhanced NMF** | **10** | **0.5837** | **202.73** | **🥇 Winner** |

### **Discovered Topic Categories**
1. **Payment & Late Fees** - Payment issues, late fees, due dates
2. **Credit Reporting** - Credit reports, bureaus (Experian, Equifax), disputes  
3. **Customer Service Calls** - Phone interactions, communication issues
4. **Account Management** - Account opening/closing, balance issues
5. **Debt Collection** - Collection agencies, debt disputes
6. **Mortgage & Loans** - Home loans, modifications, interest rates
7. **Credit Cards** - Credit card charges, purchases, balances
8. **Wells Fargo Issues** - Specific bank-related complaints
9. **Correspondence** - Letters, documentation, dispute processes
10. **Bank of America** - Banking fees, overdrafts, deposits

### **Advanced Analysis Results**
- **Topic Overlap Analysis**: Maximum Jaccard similarity of 0.82 between LDA and NMF topics
- **Parameter Optimization**: Systematic grid search improved topic coherence by 33%
- **Representative Document Analysis**: Automated identification of most relevant complaints per topic

## Repository Structure

- `/data`: Processed data files
- `/notebooks`: Jupyter notebooks with analysis
- `/results`: Output from analysis
  - `/results/models`: Saved models
  - `/results/visualizations`: Generated visualizations
- `requirements.txt`: Dependencies

## Setup and Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

See the main notebook `nlp_topic_modeling_pipeline.ipynb` for the complete analysis.

## Visualizations

Interactive LDA visualization available in `results/visualizations/lda_visualization.html`

## Author

Analysis performed on 2025-06-11
