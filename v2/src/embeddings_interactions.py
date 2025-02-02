from sentence_transformers import SentenceTransformer, util
import networkx as nx
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from email.utils import parseaddr
import seaborn as sns

from utils import save_df

def run_embeddings_chat(df):

    df.rename(columns={"Content": "Message"}, inplace=True)

    # Initialize sentence transformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Function to perform really basic cleaning of the original text
    def micro_clean_text(text):
        text = str(text)
        text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
        text = text.lower().strip()  # Lowercase and trim spaces
        return text

    # Add sender username to message for better embeddings
    df["Processed_Message"] = df.apply(lambda row: f"{row['User']}: {micro_clean_text(row['Message'])}", axis=1)

    # Identify questions based on the presence of '?'
    df["is_question"] = df["Message"].apply(lambda x: "?" in str(x))

    # Encode messages with user information
    messages = df["Processed_Message"].tolist()
    embeddings = model.encode(messages, convert_to_tensor=True)

    # Create directed graph
    G = nx.DiGraph()

    interactions = []

    # Process each message to find question-response pairs
    # i have for now estabilished 0.5 as the threshold for the similarity
    for i, row in df.iterrows():
        if row["is_question"]:  # If the message is a question
            question_user = row["User"]
            question_text = row["Processed_Message"]
            question_embedding = embeddings[i]

            # Check the next N messages for potential answers
            for j in range(i + 1, min(i + 5, len(df))):  # Look at the next 5 messages
                response_user = df.iloc[j]["User"]
                response_text = df.iloc[j]["Processed_Message"]
                response_embedding = embeddings[j]

                # Compute cosine similarity between question and response
                similarity = util.pytorch_cos_sim(question_embedding, response_embedding).item()

                if similarity > 0.5 and response_user != question_user:  # Define similarity threshold
                    G.add_edge(question_user, response_user, weight=similarity)
                    interactions.append((question_user, response_user, similarity))

    # Convert interactions to DataFrame
    interactions_df = pd.DataFrame(interactions, columns=["Question_User", "Response_User", "Similarity"])
    interactions_df = interactions_df.sort_values(by="Similarity", ascending=False)  # Sort by highest similarity

    return G, interactions_df

def draw_interaction_network(G, title="Question-Response Interaction Network"):
    """
    Draw the question-response interaction network.

    :param G: NetworkX graph object representing the interaction network.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=8, edge_color="gray", arrows=True)
    plt.title(title)
    plt.show()

def analyze_email_dialogues(df, output_folder="run_results/dialogues", dataset='clinton_emails', timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")):
    """
    Analyze email dialogues and save results.
    
    Args:
    df: DataFrame containing Clinton emails
    output_folder: Base output folder path
    dataset: Name of the dataset
    
    Returns:
    Tuple of (NetworkX Graph, DataFrame of interactions)
    """
    
    # def clean_email_address(addr):
    #     """Clean and standardize email addresses"""
    #     _, email = parseaddr(str(addr))
    #     return email.lower() if email else str(addr).lower()
    
    print("Analyzing email dialogues...")
    if 'DateSent' not in df.columns:
        df['DateSent'] = pd.Timestamp.today().normalize()
    # Create network graph
    G = nx.DiGraph()
    
    interactions = []
    
    # Process emails
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing emails"):
        sender = row['Sender']
        
        # Process recipients
        if pd.notna(row['Recipient']):
            recipients = [r for r in str(row['Recipient']).split(';')]
            
            # Add edges to graph and record interactions
            for recipient in recipients:
                # Update edge weight
                weight = G.get_edge_data(sender, recipient, {'weight': 0})['weight'] + 1
                G.add_edge(sender, recipient, weight=weight)
                
                interactions.append({
                    'sender': sender,
                    'recipient': recipient,
                    'date': row['DateSent'],
                    'subject': row['Subject'],
                    'weight': weight
                })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Save results
    save_df(interactions_df, "user_interactions.csv", output_folder, dataset, timestamp)
    
    return G, interactions_df


def visualize_dialogue_patterns(G, interactions_df, title_prefix="Email"):
    """
    Create visualizations for dialogue analysis.
    
    Args:
    G: NetworkX graph of interactions
    interactions_df: DataFrame of interactions
    title_prefix: Prefix for plot titles
    """
    # Network Graph
    plt.figure(figsize=(12, 8))
    draw_interaction_network(G, f"{title_prefix} Communication Network")
    
    # Top Participants
    plt.figure(figsize=(12, 6))
    top_senders = dict(G.out_degree(weight='weight'))
    top_senders_df = pd.DataFrame.from_dict(top_senders, orient='index', 
                                          columns=['Messages Sent']).sort_values('Messages Sent', 
                                                                              ascending=False).head(10)
    sns.barplot(data=top_senders_df, x=top_senders_df.index, y='Messages Sent')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top {title_prefix} Senders")
    plt.tight_layout()
    plt.show()
    
    # Communication Over Time
    if 'date' in interactions_df.columns:
        plt.figure(figsize=(12, 6))
        interactions_df['date'] = pd.to_datetime(interactions_df['date'])
        daily_counts = interactions_df.groupby(interactions_df['date'].dt.date).size()
        daily_counts.plot(kind='line')
        plt.title(f"{title_prefix} Communications Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Communications")
        plt.tight_layout()
        plt.show()


def analyze_communities(G):
    """
    Analyze the communities in the interaction graph using alternative algorithms.
    Returns a dictionary mapping nodes to their communities.
    """
    try:
        # Method 1: Girvan-Newman
        communities_generator = nx.community.girvan_newman(G)
        communities = next(communities_generator)
        
        # Convert the result into a dictionary node -> community_id
        partition = {}
        for community_id, community in enumerate(communities):
            for node in community:
                partition[node] = community_id
                
    except Exception as e:
        print(f"Girvan-Newman failed, using connected components: {str(e)}")
        # Fallback: Use connected components as communities
        communities = nx.connected_components(G.to_undirected())
        partition = {}
        for community_id, community in enumerate(communities):
            for node in community:
                partition[node] = community_id
    
    return partition

def draw_community_network(G, partition, title="Email Communication Network with Communities"):
    """
    Display the graph with communities highlighted by different colors.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Prepare colors for the nodes
    colors = [partition[node] for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, 
           node_color=colors,
           with_labels=True, 
           cmap=plt.cm.tab20, 
           node_size=200, 
           font_size=8, 
           edge_color="gray",
           arrows=True)
    
    plt.title(title)
    plt.show()
    
    # Print some statistics about the communities
    n_communities = len(set(partition.values()))
    print(f"\nNumber of communities found: {n_communities}")
    
    # Size of the communities
    community_sizes = {}
    for community_id in set(partition.values()):
        size = sum(1 for v in partition.values() if v == community_id)
        community_sizes[community_id] = size
    
    print("\nSize of the communities:")
    for comm_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"Community {comm_id}: {size} members")