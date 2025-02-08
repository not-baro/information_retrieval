import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer
import numpy as np

def run_sentiment_analysis(df, text_column='CleanedText', batch_size=32):
    """
    Run sentiment analysis on text data and visualize results.
    
    Args:
    df: DataFrame containing the texts
    text_column: Name of the column containing text to analyze
    batch_size: Number of texts to process at once
    
    Returns:
    DataFrame with original text and sentiment scores
    """

    
    # Initialize tokenizer and sentiment analyzer with truncation
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model=model_name,
                                tokenizer=tokenizer,
                                truncation=True,  # Enable truncation
                                max_length=512,   # Set maximum length
                                batch_size=batch_size)
    
    print("Running sentiment analysis...")
    
    # Process texts in batches
    texts = df[text_column].dropna().tolist()
    results = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_results = sentiment_analyzer(batch)
        results.extend(batch_results)
    
    # Create results DataFrame
    sentiment_df = pd.DataFrame({
        'Id': df[df[text_column].notna()]['Id'].values,
        'Text': texts,
        'Sentiment': [r['label'] for r in results],
        'Score': [r['score'] for r in results]
    })
    
    # Create visualizations
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(8, 8))  # Fixed larger size for better visibility
    sentiment_counts = sentiment_df['Sentiment'].value_counts()
    colors = {'POSITIVE': 'lightgreen', 'NEGATIVE': 'lightcoral'}
    plt.pie(sentiment_counts.values, 
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=[colors[sent] for sent in sentiment_counts.index],
            textprops={'fontsize': 14, 'fontweight': 'bold'})  # Larger and bolder text
    plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')  # Larger and bolder title
    plt.show()
    
    # 2. Score Distribution
    plt.figure(figsize=(5, 3))
    sns.histplot(data=sentiment_df, x='Score', bins=30)
    plt.title('Confidence Score Distribution')
    plt.show()
    
    # 3. Score Box Plot by Sentiment
    plt.figure(figsize=(5, 3))
    sns.boxplot(data=sentiment_df, x='Sentiment', y='Score')
    plt.title('Score Distribution by Sentiment')
    plt.show()
    
    # Print summary statistics
    print("\nSentiment Analysis Summary:")
    print("-" * 50)
    print(f"Total documents analyzed: {len(sentiment_df)}")
    print("\nSentiment Distribution:")
    print(sentiment_df['Sentiment'].value_counts().to_string())
    print("\nScore Statistics:")
    print(sentiment_df.groupby('Sentiment')['Score'].describe())
    
    return sentiment_df


