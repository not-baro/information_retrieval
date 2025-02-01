from sentence_transformers import SentenceTransformer, util
import networkx as nx
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

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

    # Store interactions
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

def save_interactions_df(dataset, output_folder, results, timestamp):
    output_folder = f"{output_folder}/{dataset}/{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "user_interactions.csv")
    results.to_csv(output_path, index=False)