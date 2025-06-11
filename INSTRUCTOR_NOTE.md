📋 **IMPORTANT NOTE FOR INSTRUCTOR/EVALUATOR**

## Dataset Setup Required

This NLP Topic Modeling project requires the original Consumer Complaints dataset to run the analysis. Due to GitHub's file size limitations (100MB), the original dataset is **NOT included** in this repository.

### 🚨 **Action Required Before Evaluation:**

1. **Download the Dataset:**
   - Visit: https://www.kaggle.com/cfpb/us-consumer-finance-complaints
   - Download `consumer_complaints.csv` (approximately 150MB)
   - Alternative: https://www.consumerfinance.gov/data-research/consumer-complaints/

2. **Place in Project Root:**
   ```
   nlp-project-IU/
   ├── consumer_complaints.csv  ← **PLACE DATASET HERE**
   ├── nlp_topic_modeling_pipeline.ipynb
   ├── README.md
   └── ...
   ```

3. **Verify Setup:**
   ```powershell
   # Check if dataset exists (Windows PowerShell)
   Test-Path "consumer_complaints.csv"
   # Should return: True
   
   # Check file size (should be ~150MB)
   (Get-Item "consumer_complaints.csv").Length / 1MB
   ```

### 📊 **What's Already Included:**

✅ **Sample Data**: `data/cleaned_data_preview.csv` (10 rows preview)  
✅ **All Results**: Complete analysis outputs in `results/` folder  
✅ **All Visualizations**: Charts and plots in `results/visualizations/`  
✅ **Trained Models**: Saved models in `results/models/`  
✅ **Analysis Summary**: JSON reports with all metrics  

### 🔄 **Alternative: View Results Without Re-running**

If you prefer to **evaluate without downloading the dataset**, you can:

1. **Open the main notebook**: `nlp_topic_modeling_pipeline.ipynb`
2. **View all outputs**: All cells show pre-computed results
3. **Check visualizations**: Available in `results/visualizations/`
4. **Review analysis**: Complete summary in `results/enhanced_nmf_analysis.json`

### 💡 **Quick Evaluation Path:**

```powershell
# 1. Clone and enter directory
git clone https://github.com/joshuarebo/nlp-project-IU.git
cd nlp-project-IU

# 2. View the main notebook (all outputs preserved)
jupyter notebook nlp_topic_modeling_pipeline.ipynb

# 3. Review analysis summary
Get-Content results/enhanced_nmf_analysis.json | ConvertFrom-Json
```

### 📞 **Contact:**
**Student**: Joshua Rebo  
**Email**: joshua.rebo@iu-study.org  
**Matriculation**: 9213334  

---
*This note ensures smooth project evaluation. The analysis demonstrates advanced NLP techniques with comprehensive results already computed and visualized.*
