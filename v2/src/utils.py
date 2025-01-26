import os
import pickle
import json
from datetime import datetime

def save_lda_model(dataset, output_folder, lda, lda_topics, lda_coherence, perplexity, topic_diversity, lda_time):
    """
    Save the LDA model, topics, and metrics in separate files.
    
    :param dataset: Name of the dataset (used to create the output folder).
    :param output_folder: Main output folder.
    :param lda: Trained LDA model.
    :param lda_topics: List of extracted topics.
    :param lda_coherence: Coherence of the LDA model.
    :param perplexity: Perplexity of the LDA model.
    :param topic_diversity: Diversity of the topics.
    :param lda_time: Execution time of the LDA model.
    """
    # Define the output folder path
    output_folder = os.path.join(output_folder, dataset)
    os.makedirs(output_folder, exist_ok=True)

    # Subfolders to organize the results
    subfolders = ["models", "topics", "metrics"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)

    # Save the LDA model
    model_path = os.path.join(output_folder, "models", "lda_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lda, f)
    print(f"Modello LDA salvato in: {model_path}")

    # Save the extracted topics in a JSON file
    topics_data = {
        "topics": lda_topics,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    topics_path = os.path.join(output_folder, "topics", "lda_topics.json")
    with open(topics_path, "w") as f:
        json.dump(topics_data, f, indent=4)
    print(f"Topic salvati in: {topics_path}")

    # Save the metrics in a JSON file
    metrics_data = {
        "coherence_score": lda_coherence,
        "perplexity": perplexity,
        "topic_diversity": topic_diversity,
        "execution_time": lda_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metrics_path = os.path.join(output_folder, "metrics", "lda_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metriche salvate in: {metrics_path}")

def load_lda_model(dataset, output_folder):
    """
    Load the LDA model, topics, and metrics from files.
    :param dataset: Name of the dataset (used to create the output folder).
    :param output_folder: Main output folder.
    :return: Loaded LDA model, topics, and metrics.
    """
    # Define the output folder path
    output_folder = os.path.join(output_folder, dataset)
    print(f"Output folder: {output_folder}")

    # Load the LDA model
    model_path = os.path.join(output_folder, "models", "lda_model.pkl")
    with open(model_path, "rb") as f:
        lda = pickle.load(f)

    # load stats and topics
    topics_path = os.path.join(output_folder, "topics", "lda_topics.json")
    with open(topics_path, "r") as f:
        topics = json.load(f)

    # load metrics
    metrics_path = os.path.join(output_folder, "metrics", "lda_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    return lda, topics, metrics