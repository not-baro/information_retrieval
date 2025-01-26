import pandas as pd
import re
import xml.etree.ElementTree as ET
import os

def preprocess_body_text(text):
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove timestamps
    text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}(?:\s?[APap][Mm])?', '', text)
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove "FW:", "RE:", "Fw:", "Re:", "FW", "RE", "Fw", and "Re"
    text = re.sub(r'\b[Ff][Ww]:?|\b[Rr][Ee]:?', '', text)
    # Remove website links including http, https, or only www
    text = re.sub(r'http\S+|https\S+|www\.\S+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove single letters cause they are probably signatures
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_enron_df(email_text):
    """
    Extracts components from a raw email text and returns them as a dictionary.
    
    Args:
    email_text (str): The raw email text.
    
    Returns:
    dict: A dictionary containing the extracted components: 'from', 'to', 'cc', and 'ExtractedBodyText'.
    """
    # Use regular expressions to extract the required fields
    from_match = re.search(r'From: (.+)', email_text)
    to_match = re.search(r'To: (.+)', email_text)
    subject_match = re.search(r'Subject: (.+)', email_text)

    cc_match = re.search(r'X-cc: (.+)', email_text)
    body_match = re.search(r'\n\n(.+)', email_text, re.DOTALL)  # Extracts the body after the headers

    # Extract the values or set to None if not found
    from_email = from_match.group(1).strip() if from_match else None
    to_email = to_match.group(1).strip() if to_match else None
    cc_email = cc_match.group(1).strip() if cc_match else None
    subject_email = preprocess_body_text(subject_match.group(1).strip()) if subject_match else None
    #body_text = body_match.group(1).strip() if body_match else None
    body_text = preprocess_body_text(body_match.group(1).strip()) if body_match else None

    return {
        'from': from_email,
        'to': to_email,
        'cc': cc_email,
        'subject': subject_email,
        'CleanedText': body_text
    }

def preprocess_fraudolent_emails(email_text):
    # Split the email using "From r" as delimiter
    fraudolent_emails_text = re.split(r"(?=From r)", email_text)

    # Extract the relevant fields from each email
    data = []
    for email in fraudolent_emails_text:
        from_match = re.search(r"From: (.+)", email)
        to_match = re.search(r"To: (.+)", email)
        date_match = re.search(r"Date: (.+)", email)
        subject_match = re.search(r"Subject: (.+)", email)
        body_match = re.search(r"(?:\n\n)(.+)", email, re.DOTALL)  # Content after two newlines

        data.append({
            "from": from_match.group(1).strip() if from_match else None,
            "to": to_match.group(1).strip() if to_match else None,
            "date": date_match.group(1).strip() if date_match else None,
            "subject": subject_match.group(1).strip() if subject_match else None,
            "body": body_match.group(1).strip() if body_match else None
        })
    
    # create a dataframe
    fraudolent_emails = pd.DataFrame(data)

    # drop na and duplicates and apply preprocess_body_text to subject and body
    fraudolent_emails.dropna(inplace=True)
    fraudolent_emails.drop_duplicates(inplace=True)
    fraudolent_emails['subject'] = fraudolent_emails['subject'].apply(preprocess_body_text)
    fraudolent_emails['body'] = fraudolent_emails['body'].apply(preprocess_body_text)

    return fraudolent_emails
    
    # Function to parse a single XML file
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []

    for post in root.findall(".//Post"):
        user = post.get("user")  # User attribute
        post_class = post.get("class")  # Class attribute
        content = post.text.strip() if post.text else ""  # Post content
        tokens = [(t.get("word"), t.get("pos")) for t in post.findall(".//t")]  # Tokens and POS
        word_count = len(tokens)  # Number of tokens (words) in the post

        # Append to data
        data.append({
            "File": os.path.basename(file_path),  # Name of the file
            "User": user,
            "Class": post_class,
            "Content": content,
            "Tokens": tokens,
            "WordCount": word_count,
            "FirstWord": tokens[0][0] if tokens else None,  # First word in the post
            "LastWord": tokens[-1][0] if tokens else None,  # Last word in the post
        })

    return data