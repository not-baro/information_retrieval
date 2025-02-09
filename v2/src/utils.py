import os
import pickle
import json
from datetime import datetime

def save_df(df, filename, output_folder, dataset, timestamp):
    """
    Save a DataFrame to a CSV file in a structured output folder.

    Args:
    df: DataFrame to be saved.
    filename: Name of the output CSV file.
    output_folder: Base output folder path.
    dataset: Name of the dataset.
    timestamp: Timestamp to create unique folder.
    """
    print(f"Saving results in: {output_folder}/{dataset}/run_{timestamp}")
    output_folder = f"{output_folder}/{dataset}/run_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False)
    print(f"Results saved in: {output_path}")

def save_lda_model(dataset, output_folder, lda, lda_topics, lda_coherence, perplexity, topic_diversity, lda_time, n_components, max_df, min_df, stop_words, max_iter, random_state, timestamp):
    """
    Save the LDA model, topics, and metrics in separate files within a timestamped subfolder.
    
    Args:
    dataset: Name of the dataset (used to create the output folder).
    output_folder: Main output folder.
    lda: Trained LDA model.
    lda_topics: List of extracted topics.
    lda_coherence: Coherence of the LDA model.
    perplexity: Perplexity of the LDA model.
    topic_diversity: Diversity of the topics.
    lda_time: Execution time of the LDA model.
    n_components: Number of topics extracted.
    max_df: Maximum document frequency for the CountVectorizer.
    min_df: Minimum document frequency for the CountVectorizer.
    stop_words: Stop words used during vectorization.
    max_iter: Maximum number of iterations for the LDA algorithm.
    random_state: Random state for reproducibility.
    """
    
    # Define the output folder path with a timestamped subfolder
    run_folder = os.path.join(output_folder, dataset, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # Subfolders to organize the results
    subfolders = ["models", "topics", "metrics"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(run_folder, subfolder), exist_ok=True)

    # Save the LDA model
    model_path = os.path.join(run_folder, "models", "lda_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lda, f)
    print(f"LDA model saved in: {model_path}")

    # Save the extracted topics in a JSON file
    topics_data = {
        "topics": lda_topics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    topics_path = os.path.join(run_folder, "topics", "lda_topics.json")
    with open(topics_path, "w") as f:
        json.dump(topics_data, f, indent=4)
    print(f"Topics saved in: {topics_path}")

    # Save the metrics and parameters in a JSON file
    metrics_data = {
        "coherence_score": lda_coherence,
        "perplexity": perplexity,
        "topic_diversity": topic_diversity,
        "execution_time": lda_time,
        "n_components": n_components,
        "max_df": max_df,
        "min_df": min_df,
        "stop_words": stop_words,
        "max_iter": max_iter,
        "random_state": random_state,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metrics_path = os.path.join(run_folder, "metrics", "lda_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved in: {metrics_path}")

