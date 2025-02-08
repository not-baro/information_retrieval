import os
import pickle
import json
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

def calculate_topic_diversity(topics):
    """
    Calculate the diversity of topics based on unique words.
    
    :param topics: List of topics, each topic is a list of words.
    :return: Topic diversity score.
    """
    unique_words = set()
    total_words = 0
    for topic in topics:
        unique_words.update(topic)
        total_words += len(topic)
    return len(unique_words) / total_words

def run_lda(texts, n_components=3, max_df=0.95, min_df=2, stop_words='english', max_iter=10, random_state=42):
    """
    Run LDA on a given list of texts with specified parameters.

    :param texts: List of preprocessed text documents.
    :param n_components: Number of topics to extract.
    :param max_df: Maximum document frequency for the CountVectorizer.
    :param min_df: Minimum document frequency for the CountVectorizer.
    :param stop_words: Stop words to remove during vectorization.
    :param max_iter: Maximum number of iterations for the LDA algorithm.
    :param random_state: Random state for reproducibility.
    :return: LDA model, topics, coherence score, perplexity, topic diversity, and execution time.
    """
    start_time = time.time()
    
    # Step 1: Convert text to bag-of-words
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    dtm = vectorizer.fit_transform(texts)
    
    # Step 2: Initialize and fit LDA model
    lda = LatentDirichletAllocation(n_components=n_components, random_state=random_state, max_iter=max_iter, learning_method='online')
    
    with tqdm(total=lda.max_iter, desc="Fitting LDA", unit="iteration") as pbar:
        for _ in range(lda.max_iter):
            lda.partial_fit(dtm)
            pbar.update(1)
    
    lda_time = time.time() - start_time
    
    # Step 3: Extract topics and evaluate coherence
    lda_topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]] for topic in lda.components_]
    dictionary = Dictionary([text.split() for text in texts])
    corpus = [dictionary.doc2bow(text.split()) for text in texts]
    lda_coherence = CoherenceModel(topics=lda_topics, texts=[text.split() for text in texts], dictionary=dictionary, coherence='c_v').get_coherence()
    
    # Calculate Perplexity
    perplexity = lda.perplexity(dtm)
    
    # Calculate Topic Diversity
    topic_diversity = calculate_topic_diversity(lda_topics)
    
    # Print Statistics
    print(f"LDA Execution Time: {lda_time:.2f}s")
    print(f"LDA Coherence Score: {lda_coherence:.4f}")
    print(f"LDA Perplexity: {perplexity:.4f}")
    print(f"Topic Diversity: {topic_diversity:.4f}")
    print(f"LDA Topics:")
    for topic in lda_topics:
        print("Topic n " + str(lda_topics.index(topic)) + ": " + str(topic))
    
    return lda, vectorizer, n_components, lda_topics, lda_coherence, perplexity, topic_diversity, lda_time


def get_topic_assignments_lda(df, lda, vectorizer, lda_texts, n_components):
# Get topic assignments for each document
# lda.transform() returns document-topic distribution matrix
    doc_topic_dist = lda.transform(vectorizer.transform(lda_texts))

    # Get the dominant topic for each document by taking argmax
    # This gives us the topic number with highest probability for each doc
    dominant_topics = doc_topic_dist.argmax(axis=1)

    # Create DataFrame with document IDs and their dominant topics
    # We use clinton_emails['Id'] that corresponds to non-null LdaText entries
    topic_assignments = pd.DataFrame({
        'Id': df[df['LdaText'].notna()]['Id'].values,
        'Dominant_Topic': dominant_topics
    })

    # Plot the distribution of topics
    plt.figure(figsize=(10, 6))
    plt.hist(dominant_topics, bins=n_components, edgecolor='k', alpha=0.7)
    plt.xlabel('Topic')
    plt.ylabel('Number of Documents')
    plt.title('Distribution of Topics')
    plt.xticks(range(n_components))
    plt.show()

    return topic_assignments
def save_bertopic_model(dataset, output_folder, results, timestamp):
    """
    Save the BERTopic model, topics, and metrics in separate files within a timestamped subfolder.
    
    :param dataset: Name of the dataset (used to create the output folder).
    :param output_folder: Main output folder.
    :param results: Dictionary containing model, topics, topic_info, and metrics from run_bertopic.
    """
    
    # Define the output folder path with a timestamped subfolder
    run_folder = os.path.join(output_folder, dataset, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # Subfolders to organize the results
    subfolders = ["models", "topics", "metrics", "info"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(run_folder, subfolder), exist_ok=True)

    # Save the BERTopic model
    model_path = os.path.join(run_folder, "models", "bertopic_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(results['model'], f)
    print(f"BERTopic model saved in: {model_path}")

    # Save the top topics
    topics_data = {
        "top_topics": {str(topic_id): words for topic_id, words in results['top_topics']},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    topics_path = os.path.join(run_folder, "topics", "bertopic_topics.json")
    with open(topics_path, "w") as f:
        json.dump(topics_data, f, indent=4)
    print(f"Top topics saved in: {topics_path}")

    # Save topic info (excluding Representative_Docs)
    topic_info = results['topic_info'].copy()
    for info in topic_info:
        if 'Representative_Docs' in info:
            del info['Representative_Docs'] # we don't need this
    
    info_path = os.path.join(run_folder, "info", "topic_info.json")
    with open(info_path, "w") as f:
        json.dump(topic_info, f, indent=4)
    print(f"Topic info saved in: {info_path}")

    # Save all metrics
    metrics_data = {
        **results['metrics'],  # Unpack all metrics
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metrics_path = os.path.join(run_folder, "metrics", "bertopic_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved in: {metrics_path}")

def run_bertopic(texts, n_topics=None, top_n_topics=5):
    """
    Run BERTopic on a given list of texts and reduce the number of topics.

    :param texts: List of preprocessed text documents.
    :param n_topics: Desired number of topics after reduction. If None, no reduction is applied.
    :param top_n_topics: Number of top topics to display.
    :return: Dictionary containing model, topics, and BERTopic-specific metrics.
    """
    # Start timing
    start_time = time.time()
    
    # Initialize BERTopic
    bertopic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)

    # Fit the model on the preprocessed texts
    topics, probabilities = bertopic_model.fit_transform(texts)
    
    # Calculate execution time
    bert_time = time.time() - start_time

    # Reduce the number of topics if specified
    if n_topics is not None:
        bertopic_model.reduce_topics(texts, nr_topics=n_topics)

    # Get the topics and their top words
    bertopic_topics = bertopic_model.get_topics()
    
    # Get top N topics
    top_topics = list(bertopic_topics.items())[:top_n_topics]
    
    # Get topic information
    topic_info = bertopic_model.get_topic_info()
    
    # Calculate quality metrics
    try:
        # Topic Similarity Matrix (how similar topics are to each other)
        similarity_matrix = bertopic_model.get_topic_similarity()
        avg_similarity = float(similarity_matrix.mean())
        
        # Calculate custom topic diversity (ratio of unique words across topics)
        all_words = set()
        total_words = 0
        for topic_id, words in bertopic_topics.items():
            if topic_id != -1:  # Skip outlier topic
                topic_words = [word for word, _ in words]
                all_words.update(topic_words)
                total_words += len(topic_words)
        topic_diversity = len(all_words) / total_words if total_words > 0 else 0
        
    except Exception as e:
        print(f"Warning: Some metrics couldn't be calculated: {str(e)}")
        avg_similarity = topic_diversity = None

    # Calculate representative metrics using topic_info
    total_docs = topic_info['Count'].sum()
    outliers = topic_info[topic_info['Topic'] == -1]['Count'].iloc[0] if -1 in topic_info['Topic'].values else 0
    
    metrics = {
        'execution_time': bert_time,
        'n_topics': len(bertopic_topics),
        'n_reduced_topics': n_topics,
        'largest_topic_size': int(topic_info['Count'].max()),
        'smallest_topic_size': int(topic_info['Count'].min()),
        'avg_topic_size': float(topic_info['Count'].mean()),
        'total_documents': int(total_docs),
        'n_outliers': int(outliers),
        'outliers_ratio': float(outliers / total_docs) if total_docs > 0 else 0,
        # Quality metrics
        'topic_diversity': float(topic_diversity) if topic_diversity is not None else None,
        'avg_topic_similarity': avg_similarity if avg_similarity is not None else None
    }
    
    # Print Statistics
    print(f"\nBERTopic Model Statistics:")
    print(f"Execution Time: {metrics['execution_time']:.2f}s")
    print(f"Number of Topics: {metrics['n_topics']}")
    print(f"Number of Documents: {metrics['total_documents']}")
    print(f"Documents in Outlier Topic: {metrics['n_outliers']} ({metrics['outliers_ratio']:.2%})")
    
    # print(f"\nQuality Metrics:")
    # if metrics['topic_diversity'] is not None:
    #     print(f"Topic Diversity: {metrics['topic_diversity']:.4f}")
    # if metrics['avg_topic_similarity'] is not None:
    #     print(f"Average Topic Similarity: {metrics['avg_topic_similarity']:.4f}")
    
    print(f"\nTop {top_n_topics} Topics:")
    for topic_id, words in top_topics:
        print(f"Topic {topic_id}: {words}")

    return {
        'model': bertopic_model,
        'top_topics': top_topics,
        'probabilities': probabilities,
        'topic_info': topic_info.to_dict('records'),
        'metrics': metrics
    }

def visualize_bertopics(bertopic_model, top_n_words=10, figsize=(20, 10)):
    """
    Visualize topics using both bar charts and word clouds.
    
    :param bertopic_model: Trained BERTopic model
    :param top_n_words: Number of top words to show for each topic
    :param figsize: Size of the figure (width, height)
    """
    
    # Get topics and their info
    topics = bertopic_model.get_topics()
    topic_info = bertopic_model.get_topic_info()
    
    # Filter out the outlier topic (-1)
    topics = {k: v for k, v in topics.items() if k != -1}
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=figsize)
    
    # # 1. Bar Chart of Topic Sizes
    # plt.subplot(1, 2, 1)
    # # Filter out outlier topic for visualization
    # topic_sizes = topic_info[topic_info['Topic'] != -1]
    # plt.bar(topic_sizes['Topic'].astype(str), topic_sizes['Count'], color='skyblue')
    # plt.title('Topic Sizes')
    # plt.xlabel('Topic')
    # plt.ylabel('Number of Documents')
    # plt.xticks(rotation=45)
    
    # 2. Word Clouds for Top Topics
    plt.subplot(1, 2, 2)
    
    # Combine words from all topics for the word cloud
    all_words = {}
    for topic_id, word_scores in topics.items():
        for word, score in word_scores[:top_n_words]:
            # Use the absolute value of the score as weight
            all_words[word] = abs(float(score))
    
    # Create and display the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(all_words)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Topics Word Cloud')
    
    plt.tight_layout()
    plt.show()
    
    # Create individual word clouds for each topic
    n_topics = len(topics)
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5*n_rows))
    
    for idx, (topic_id, word_scores) in enumerate(topics.items(), 1):
        # Create word frequency dict for this topic
        topic_words = {word: abs(float(score)) for word, score in word_scores[:top_n_words]}
        
        plt.subplot(n_rows, n_cols, idx)
        
        # Create and display the word cloud for this topic
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color='white',
            colormap='viridis'
        ).generate_from_frequencies(topic_words)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_id}')
    
    plt.tight_layout()
    plt.show()
    
    # Bar charts for individual topics
    plt.figure(figsize=(20, 5*n_rows))
    
    for idx, (topic_id, word_scores) in enumerate(topics.items(), 1):
        plt.subplot(n_rows, n_cols, idx)
        
        words, scores = zip(*word_scores[:top_n_words])
        plt.barh(range(len(words)), [abs(score) for score in scores], color='skyblue')
        plt.yticks(range(len(words)), words)
        plt.title(f'Topic {topic_id}')
        plt.xlabel('Score')
    
    plt.tight_layout()
    plt.show()

def visualize_lda_topics(lda_model, vectorizer, lda_topics, n_top_words=10, figsize=(20, 10)):
    """
    Visualize LDA topics using both bar charts and word clouds.
    
    :param lda_model: Trained LDA model
    :param vectorizer: CountVectorizer used for the LDA model
    :param lda_topics: List of topics with their words
    :param n_top_words: Number of top words to show for each topic
    :param figsize: Size of the figure (width, height)
    """
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    
    feature_names = vectorizer.get_feature_names_out()
    n_topics = len(lda_model.components_)
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=figsize)
    
    # # 1. Bar Chart of Topic Distributions
    # plt.subplot(1, 2, 1)
    # # Calculate average topic distributions
    # topic_distributions = lda_model.transform(vectorizer.transform([' '.join(topic) for topic in lda_topics]))
    # avg_topic_dist = topic_distributions.mean(axis=0)
    # plt.bar(range(n_topics), avg_topic_dist, color='skyblue')
    # plt.title('Average Topic Distribution')
    # plt.xlabel('Topic')
    # plt.ylabel('Average Weight')
    # plt.xticks(range(n_topics))
    
    # 2. Word Cloud for All Topics
    plt.subplot(1, 2, 2)
    
    # Combine words from all topics for the word cloud
    all_words = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words-1:-1]
        for idx in top_features_ind:
            word = feature_names[idx]
            weight = topic[idx]
            all_words[word] = all_words.get(word, 0) + abs(float(weight))
    
    # Create and display the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(all_words)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Topics Word Cloud')
    
    plt.tight_layout()
    plt.show()
    
    # Individual word clouds for each topic
    n_cols = 3
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5*n_rows))
    
    for topic_idx, topic in enumerate(lda_model.components_):
        plt.subplot(n_rows, n_cols, topic_idx + 1)
        
        # Create word frequency dict for this topic
        top_features_ind = topic.argsort()[:-n_top_words-1:-1]
        topic_words = {
            feature_names[i]: abs(float(topic[i])) 
            for i in top_features_ind
        }
        
        # Create and display the word cloud for this topic
        wordcloud = WordCloud(
            width=400,
            height=200,
            background_color='white',
            colormap='viridis'
        ).generate_from_frequencies(topic_words)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {topic_idx}')
    
    plt.tight_layout()
    plt.show()
    
    # Bar charts for individual topics
    plt.figure(figsize=(20, 5*n_rows))
    
    for topic_idx, topic in enumerate(lda_model.components_):
        plt.subplot(n_rows, n_cols, topic_idx + 1)
        
        top_features_ind = topic.argsort()[:-n_top_words-1:-1]
        top_words = [feature_names[i] for i in top_features_ind]
        top_weights = [topic[i] for i in top_features_ind]
        
        plt.barh(range(len(top_words)), [abs(w) for w in top_weights], color='skyblue')
        plt.yticks(range(len(top_words)), top_words)
        plt.title(f'Topic {topic_idx}')
        plt.xlabel('Weight')
    
    plt.tight_layout()
    plt.show()
