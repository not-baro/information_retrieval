import pandas as pd
import os
import spacy
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json
from transformers import pipeline
from utils import save_df

def run_ner_spacy(texts, model_name="en_core_web_sm"):
    """
    Run Named Entity Recognition using spaCy.
    
    Args:
    texts: List of preprocessed text documents
    model_name: Name of the spaCy model to use
    
    Returns:
    Dictionary containing entities and metrics
    """
    start_time = time.time()
    
    # Load spaCy model
    nlp = spacy.load(model_name)
    
    # Process texts and extract entities
    entities = []
    for doc_id, text in tqdm(enumerate(texts), desc="Processing with spaCy"):
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({
                'doc_id': doc_id,
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    
    # Create DataFrame with results
    df_entities = pd.DataFrame(entities)
    
    # Calculate metrics
    execution_time = time.time() - start_time
    metrics = {
        'execution_time': execution_time,
        'n_documents': len(texts),
        'n_entities': len(entities),
        'avg_entities_per_doc': len(entities) / len(texts),
        'unique_entity_types': len(df_entities['label'].unique()),
        'entity_type_distribution': df_entities['label'].value_counts().to_dict()
    }
    
    return {
        'entities': df_entities,
        'metrics': metrics
    }

def run_ner_bert(texts, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    """
    Run Named Entity Recognition using BERT.
    
    Args:
    texts: List of preprocessed text documents
    model_name: Name of the BERT model to use
    
    Returns:
    Dictionary containing entities and metrics
    """
    start_time = time.time()
    
    # Initialize BERT pipeline
    ner_pipeline = pipeline("ner", model=model_name)
    
    # Process texts and extract entities
    entities = []
    for doc_id, text in tqdm(enumerate(texts), desc="Processing with BERT"):
        # Handle maximum length for BERT
        if len(text) > 512:
            text = text[:512]
        
        try:
            ents = ner_pipeline(text)
            for ent in ents:
                entities.append({
                    'doc_id': doc_id,
                    'text': ent['word'],
                    'label': ent['entity'],
                    'score': ent['score'],
                    'start': ent['start'],
                    'end': ent['end']
                })
        except Exception as e:
            print(f"Error processing document {doc_id}: {str(e)}")
    
    # Create DataFrame with results
    df_entities = pd.DataFrame(entities)
    
    # Calculate metrics
    execution_time = time.time() - start_time
    metrics = {
        'execution_time': execution_time,
        'n_documents': len(texts),
        'n_entities': len(entities),
        'avg_entities_per_doc': len(entities) / len(texts),
        'unique_entity_types': len(df_entities['label'].unique()),
        'entity_type_distribution': df_entities['label'].value_counts().to_dict(),
        'avg_confidence': df_entities['score'].mean()
    }
    
    return {
        'entities': df_entities,
        'metrics': metrics
    }

def save_ner_results(dataset, output_folder, model_type, results, timestamp):
    """
    Save NER results in a structured format.
    
    Args:
    dataset: Name of the dataset
    output_folder: Base output folder
    model_type: 'spacy' or 'bert'
    results: Dictionary containing entities and metrics
    """
    print(f"Saving {model_type} results...")
    run_folder = os.path.join(output_folder, dataset, f"run_{timestamp}", model_type)
    
    # Create necessary folders
    for subfolder in ["entities", "metrics"]:
        os.makedirs(os.path.join(run_folder, subfolder), exist_ok=True)
    
    # # Save entities
    # print("Saving entities...")
    # entities_path = os.path.join(run_folder, "entities", f"{model_type}_entities.csv")
    # results['entities'].to_csv(entities_path, index=False)

    save_df(results['entities'], os.path.join(run_folder, "entities", f"{model_type}_entities.csv"))
    
    # Convert numpy types to Python native types for JSON serialization
    metrics = results['metrics'].copy()
    for key, value in metrics.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (like entity_type_distribution)
            metrics[key] = {k: float(v) if hasattr(v, 'dtype') else v 
                          for k, v in value.items()}
        else:
            # Handle direct numpy values
            metrics[key] = float(value) if hasattr(value, 'dtype') else value
    
    # Save metrics
    print("Saving metrics...")
    metrics_path = os.path.join(run_folder, "metrics", f"{model_type}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved in: {run_folder}")
    return run_folder

def visualize_ner_results(results, title_prefix=""):
    """
    Visualize NER results using various plots.
    
    Args:
    results: Dictionary containing entities and metrics
    title_prefix: Prefix for plot titles (e.g., "spaCy" or "BERT")
    """
    df = results['entities']
    
    # 1. Entity Type Distribution
    plt.figure(figsize=(15, 6))
    sns.countplot(data=df, y='label', order=df['label'].value_counts().index)
    plt.title(f'{title_prefix} Entity Type Distribution')
    plt.xlabel('Count')
    plt.ylabel('Entity Type')
    plt.tight_layout()
    plt.show()
    
    # 2. Top Entities per Type
    plt.figure(figsize=(20, 4 * (len(df['label'].unique()) + 1) // 2))
    for i, label in enumerate(df['label'].unique(), 1):
        plt.subplot((len(df['label'].unique()) + 1) // 2, 2, i)
        top_entities = df[df['label'] == label]['text'].value_counts().head(10)
        sns.barplot(x=top_entities.values, y=top_entities.index)
        plt.title(f'Top 10 {label} Entities')
        plt.xlabel('Count')
    plt.tight_layout()
    plt.show()
    
    # 3. If BERT results, show confidence distribution
    if 'score' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='score', bins=50)
        plt.title(f'{title_prefix} Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.show()
