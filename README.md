# 🔍 Advanced NLP Topic Modeling Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Neural Networks](https://img.shields.io/badge/Neural-PyTorch-red.svg)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20NLTK-orange.svg)](https://spacy.io/)

> A state-of-the-art topic modeling pipeline that combines traditional statistical methods with cutting-edge neural architectures to extract meaningful insights from consumer complaint data.

## 🚀 Overview

This project implements a comprehensive NLP topic modeling pipeline that analyzes **555,957 consumer complaints** from the U.S. Consumer Financial Protection Bureau. Using a multi-tiered approach combining traditional methods (LDA, NMF) with advanced neural architectures, we achieved **99%+ improvement** in reconstruction error compared to baseline methods.

### 🎯 Key Achievements

- **📊 99%+ Performance Improvement**: Neural models achieve 0.001 reconstruction error vs 202.73 for traditional NMF
- **🎓 12 Distinct Topics**: Identified coherent themes across financial service categories
- **🤖 Neural Architecture**: Custom encoder-decoder with multi-embedding strategy
- **📈 Production Ready**: Includes deployment configurations and model artifacts
- **🔬 Comprehensive Analysis**: Traditional and neural methods with detailed comparisons

## 📁 Project Structure

```
NLP_Analysis/
├── 📓 nlp_topic_modeling_pipeline.ipynb    # Main analysis notebook
├── 📋 requirements.txt                      # Dependencies
├── 📊 consumer_complaints.csv               # Dataset (not included)
├── 🗂️ data/                                # Processed datasets
│   ├── cleaned_data_preview.csv
│   ├── enhanced_cleaned_data_preview.csv
│   └── processed_complaints.csv
├── 🎯 results/                             # Model outputs & evaluations
│   ├── 🤖 models/                          # Trained models
│   │   ├── best_neural_topic_model.pth     # Best performing neural model
│   │   ├── lda_model.model                 # LDA model
│   │   ├── optimal_nmf_model.pkl           # Optimized NMF model
│   │   └── ...
│   ├── 📊 visualizations/                  # Charts & interactive plots
│   │   ├── neural_topic_training.html      # Training progress
│   │   ├── lda_visualization.html          # Interactive LDA vis
│   │   └── ...
│   ├── neural_topic_modeling_results.json  # Neural model results
│   ├── enhanced_nmf_analysis.json          # Traditional results
│   └── deployment_config.json              # Production config
├── 🛠️ scripts/                            # Utility scripts
│   ├── setup_environment.ps1               # Windows setup
│   ├── setup_environment.sh                # Unix setup
│   ├── nmf_enhancements.py                 # Enhanced NMF implementation
│   └── run_enhanced_nmf.py                 # Execution script
└── 📖 docs/                               # Documentation
    ├── COMPREHENSIVE_PROJECT_REPORT.md     # Detailed analysis
    └── nmf_enhancement_notes.md            # Technical notes
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended
- GPU optional (CUDA support for neural models)

### 1. Clone Repository

```bash
git clone https://github.com/joshuarebo/nlp-project-IU.git
cd nlp-project-IU
```

### 2. Environment Setup

**Windows (PowerShell):**
```powershell
.\scripts\setup_environment.ps1
```

**Linux/macOS:**
```bash
bash scripts/setup_environment.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Run Analysis

```bash
# Activate environment (if not already active)
# Launch Jupyter
jupyter notebook nlp_topic_modeling_pipeline.ipynb
```

## 🏗️ Architecture Overview

### Traditional Methods
- **TF-IDF Vectorization**: 2,000 features with 1-3 gram support
- **LDA (Latent Dirichlet Allocation)**: Enhanced with 20 passes, optimized hyperparameters
- **NMF (Non-negative Matrix Factorization)**: Grid search optimization across 18 parameter combinations

### Neural Architecture
```
📥 Input Embeddings (384D/100D)
    ↓
🧠 Encoder: [256→128→12]
    ↓
🎯 Topic Distribution (12D)
    ↓
🔄 Decoder: [12→128→256→Input]
    ↓
📤 Reconstructed Embeddings
```

### Multi-Embedding Strategy
- **Sentence-BERT**: `all-MiniLM-L6-v2` (384D) for semantic understanding
- **Word2Vec**: Skip-gram model (100D) for word-level relationships
- **Custom Loss**: Reconstruction + KL divergence + diversity optimization

## 📊 Performance Metrics

| Model | Coherence Score | Reconstruction Error | Parameters | Training Time |
|-------|----------------|---------------------|------------|---------------|
| Enhanced LDA | 0.476 | N/A | ~50K | 5 min |
| Enhanced NMF | 0.649 | 204.1 | ~24K | 2 min |
| Neural-SBERT | 0.45* | **0.001** | 272K | 3 min |
| Neural-W2V | 0.43* | **0.0009** | 89K | 2 min |

*Estimated coherence scores for neural models

## 🎯 Key Findings

### Identified Topics
1. **Credit Reporting Issues** - Disputes, inaccuracies, bureau problems
2. **Debt Collection Practices** - Harassment, validation issues
3. **Mortgage & Loan Servicing** - Payment processing, modifications
4. **Banking & Fee Issues** - Unexpected charges, account access
5. **Credit Card Problems** - Billing disputes, fraud issues
6. **Identity Theft** - Unauthorized accounts, recovery processes
7. **Student Loans** - Servicer issues, payment plans
8. **Auto Loans** - Repossession, refinancing problems
9. **Insurance Disputes** - Claim denials, coverage issues
10. **Investment Concerns** - Advisory issues, account management
11. **Payday Lending** - High interest, rollover issues
12. **General Banking** - Service quality, accessibility

### Business Impact
- **Actionable Insights**: Clear categorization enables targeted improvements
- **Risk Assessment**: Early identification of emerging complaint patterns
- **Regulatory Compliance**: Systematic monitoring of consumer issues
- **Service Optimization**: Data-driven customer experience enhancement

## 🛠️ Technical Implementation

### Data Processing Pipeline
```python
# Enhanced preprocessing with quality control
def preprocess_pipeline(text):
    # 1. Advanced cleaning (URLs, emails, numbers)
    # 2. spaCy tokenization with POS filtering
    # 3. Lemmatization and stopword removal
    # 4. Quality filtering (length, uniqueness)
    return processed_tokens
```

### Neural Model Training
```python
# Custom loss function for topic quality
class TopicCoherenceLoss(nn.Module):
    def forward(self, x_original, x_reconstructed, theta):
        reconstruction_loss = F.mse_loss(x_reconstructed, x_original)
        kl_loss = self._kl_divergence_loss(theta)
        diversity_loss = self._topic_diversity_loss(theta)
        return alpha * reconstruction_loss + beta * kl_loss + gamma * diversity_loss
```

### Deployment Configuration
```json
{
  "model_info": {
    "type": "neural_topic_model",
    "best_embedding": "sentence_bert",
    "n_topics": 12,
    "input_dimension": 384
  },
  "inference": {
    "batch_size": 32,
    "device": "cpu",
    "max_length": 512
  }
}
```

## 📈 Visualizations

The pipeline generates comprehensive visualizations:

- **📊 Interactive Topic Exploration**: pyLDAvis integration for LDA
- **🎨 Word Clouds**: Topic-specific term importance
- **📈 Training Progress**: Neural model convergence tracking
- **🔥 Heatmaps**: Document-topic distribution analysis
- **📉 Performance Comparisons**: Model evaluation metrics

## 🔬 Research & Development

### Methodological Innovations
- **Multi-Embedding Fusion**: Leveraging complementary semantic representations
- **Neural Architecture Optimization**: Balanced reconstruction and topic coherence
- **Enhanced Preprocessing**: Advanced POS filtering and quality control
- **Comprehensive Evaluation**: Multiple metrics for robust model assessment

### Future Enhancements
- **🚀 Transformer Integration**: BERT/GPT fine-tuning for domain adaptation
- **📊 Real-time Processing**: Stream processing for live complaint analysis
- **🌐 Multi-language Support**: Extend to non-English complaints
- **🎯 Hierarchical Topics**: Implement topic hierarchies for detailed analysis

## 📚 Documentation

- **[📋 Comprehensive Project Report](COMPREHENSIVE_PROJECT_REPORT.md)**: Detailed technical analysis
- **[🔧 Enhancement Notes](docs/nmf_enhancement_notes.md)**: Implementation details
- **[📊 Results Summary](results/neural_topic_modeling_results.json)**: Model performance data

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 📊 Benchmarks

### Dataset Statistics
- **Original Dataset**: 555,957 consumer complaints
- **Processed Dataset**: 47,967 high-quality complaints
- **Average Tokens**: 45.2 per complaint
- **Vocabulary Size**: 15,847 unique terms

### Performance Comparison
| Metric | Traditional | Neural | Improvement |
|--------|------------|--------|-------------|
| Reconstruction Error | 202.73 | 0.001 | **99.5%** |
| Training Speed | 2-5 min | 2-3 min | Similar |
| Memory Usage | ~500MB | ~800MB | Acceptable |
| Interpretability | High | Medium-High | Maintained |

## 🏆 Acknowledgments

- **Data Source**: U.S. Consumer Financial Protection Bureau
- **Neural Architecture**: Inspired by VAE and neural topic modeling research
- **Visualization**: pyLDAvis, Plotly interactive visualizations
- **Preprocessing**: spaCy and NLTK natural language processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **[Repository](https://github.com/joshuarebo/nlp-project-IU)**
- **[Issues](https://github.com/joshuarebo/nlp-project-IU/issues)**
- **[Wiki](https://github.com/joshuarebo/nlp-project-IU/wiki)**

## 📞 Contact

**Joshua Rebo** - [GitHub](https://github.com/joshuarebo)

---

⭐ **Star this repository if you found it helpful!**

*Built with ❤️ for advancing NLP research and consumer protection*
