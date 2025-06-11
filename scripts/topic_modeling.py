#!/usr/bin/env python
"""
NLP-Based Topic Modeling Pipeline

This script implements a complete topic modeling pipeline on consumer complaint data.
For the full interactive analysis, see the Jupyter notebook.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from datetime import datetime

# NLP libraries
import nltk
from nltk.corpus import stopwords
import spacy
from gensim import corpora
from gensim.models import Word2Vec, LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud
from tqdm import tqdm

def main():
    print("Starting NLP Topic Modeling Pipeline...")
    
    # Create directories for outputs
    output_dirs = ['data', 'results', 'results/visualizations', 'results/models']
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    # 1. Load the consumer complaints dataset
    print("Loading dataset...")
    df = pd.read_csv('consumer_complaints.csv', low_memory=False)
    print(f"Dataset shape: {df.shape}")
    
    # Identify the complaint narrative column
    narrative_col = 'consumer_complaint_narrative'
    if narrative_col not in df.columns:
        possible_cols = [col for col in df.columns if 'narrative' in col.lower() or 'complaint' in col.lower()]
        if possible_cols:
            narrative_col = possible_cols[0]
        else:
            raise ValueError("Could not find a column containing complaint narratives")
    
    # 2. Filter out rows with empty narratives
    df_filtered = df.dropna(subset=[narrative_col])
    
    # Take a subset for faster processing
    subset_size = min(50000, df_filtered.shape[0])
    df_subset = df_filtered.sample(subset_size, random_state=42) if df_filtered.shape[0] > subset_size else df_filtered
    
    print(f"Using {df_subset.shape[0]} complaints for analysis")
    
    # 3. Text preprocessing functions
    stop_words = set(stopwords.words('english'))
    domain_stops = {'xxxx', 'xx', 'xxx', 'x', 'xx/xx/xxxx'}
    stop_words.update(domain_stops)
    
    def clean_text(text):
        """Clean the text by removing punctuation, numbers, and special characters"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers, punctuation, and special characters
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
        text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
        
        return text.strip()
    
    def lemmatize_text(text):
        """Tokenize, remove stopwords, and lemmatize text using spaCy"""
        if not text:
            return []
        
        # Process with spaCy
        doc = nlp(text)
        
        # Lemmatize and filter tokens
        tokens = [token.lemma_ for token in doc 
                if token.lemma_.lower() not in stop_words 
                and len(token.lemma_) > 1  # Remove single-character tokens
                and not token.is_punct 
                and not token.is_space
                and not token.like_num]  # Remove numbers (backup)
        
        return tokens
    
    # Apply preprocessing
    print("Preprocessing complaint narratives...")
    tqdm.pandas(desc="Processing complaints")
    df_subset['cleaned_text'] = df_subset[narrative_col].progress_apply(clean_text)
    df_subset['tokens'] = df_subset['cleaned_text'].progress_apply(lemmatize_text)
    
    # Filter out documents with very few tokens
    min_tokens = 5
    df_processed = df_subset[df_subset['tokens'].apply(len) >= min_tokens].reset_index(drop=True)
    print(f"Final dataset shape: {df_processed.shape}")
    
    # Create a joined version of tokens for some models
    df_processed['processed_text'] = df_processed['tokens'].apply(lambda x: ' '.join(x))
    
    # 4. Save a preview of the cleaned dataset
    df_preview = df_processed[['processed_text', 'tokens']].head(10)
    df_preview.to_csv('data/cleaned_data_preview.csv', index=False)
    
    # 5. TF-IDF Vectorization
    print("Performing TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=10,
        max_features=1000,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_processed['processed_text'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # 6. Word2Vec Vectorization
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=df_processed['tokens'],
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        epochs=10,
        seed=42
    )
    
    print(f"Word2Vec model trained with {len(w2v_model.wv.index_to_key)} words")
    
    # 7. LDA Topic Modeling
    print("Performing LDA topic modeling...")
    id2word = corpora.Dictionary(df_processed['tokens'])
    corpus = [id2word.doc2bow(doc) for doc in df_processed['tokens']]
    
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=8,
        passes=10,
        alpha='auto',
        eta='auto',
        random_state=42
    )
    
    print("LDA Topics:")
    topics = lda_model.print_topics(num_words=10)
    for topic_id, topic in enumerate(topics):
        words = re.findall(r'"([^"]*)"', topic[1])
        print(f"Topic {topic_id+1}: {', '.join(words)}")
    
    # 8. LDA Visualization
    print("Generating LDA visualization...")
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(lda_vis, 'results/visualizations/lda_visualization.html')
    
    # 9. NMF Topic Modeling
    print("Performing NMF topic modeling...")
    nmf_model = NMF(
        n_components=8,
        random_state=42,
        max_iter=1000,
        alpha=0.1,
        l1_ratio=0.5
    )
    
    nmf_result = nmf_model.fit_transform(tfidf_matrix)
    
    # Get the top words for each topic
    def get_top_words_nmf(model, feature_names, n_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append((topic_idx, top_words))
        return topics
    
    print("NMF Topics:")
    top_words = 10
    nmf_topics = get_top_words_nmf(nmf_model, tfidf_feature_names, top_words)
    for topic_id, words in nmf_topics:
        print(f"Topic {topic_id+1}: {', '.join(words)}")
    
    # 10. Save results and models
    print("Saving results and models...")
    
    # Save the final processed dataset
    df_processed.to_csv('data/processed_complaints.csv', index=False)
    
    # Save models
    with open('results/models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    with open('results/models/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    w2v_model.save('results/models/word2vec_model.model')
    
    with open('results/models/doc_vectors.pkl', 'wb') as f:
        pickle.dump(None, f)  # Placeholder, calculate as needed
    
    lda_model.save('results/models/lda_model.model')
    
    with open('results/models/nmf_model.pkl', 'wb') as f:
        pickle.dump(nmf_model, f)
    
    with open('results/models/nmf_result.pkl', 'wb') as f:
        pickle.dump(nmf_result, f)
    
    print("Pipeline completed successfully!")
    print("See the Jupyter notebook for more detailed analysis and visualizations.")

if __name__ == "__main__":
    main()
