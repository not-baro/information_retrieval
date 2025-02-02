import pandas as pd
import re
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def preprocess_body_text(text):
    """
    Preprocesses email body text by removing various unwanted elements.
    
    Args:
    text (str): The input text to be preprocessed
    
    Returns:
    str: The cleaned text
    """
    # Remove email attachment markers and related content
    text = re.sub(r'_secatt_.*?(?=\s|$)', '', text)
    # Remove content type declarations and encoding information
    text = re.sub(r'content[-_]?type\s*:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'content\s*type\s*:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'charset\S+', '', text)
    text = re.sub(r'contenttransferencoding\s+\S+', '', text)
    text = re.sub(r'contentdisposition\s+\S+', '', text)
    # Remove base64 and other encoding markers
    text = re.sub(r'base64\s+\S+', '', text)
    text = re.sub(r'quotedprintable\s+', '', text)
    # Remove file attachments and names
    text = re.sub(r'filename\S+', '', text)
    text = re.sub(r'name\S+', '', text)
    # Remove encoded special characters (like 2e, 3a, etc.)
    text = re.sub(r'(?<=\s)[\da-f]{2}(?=\s|$)', '', text)
    text = text.replace('2e', '.').replace('3a', ':').replace('2c', ',').replace('40', '@')
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove timestamps
    text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}:\d{2}(?:\s?[APap][Mm])?', '', text)
    # Remove HTML tags and their content
    text = re.sub(r'<[^>]+>', '', text)
    # Remove application and content type markers
    text = re.sub(r'application\S+', '', text)
    text = re.sub(r'octetstream', '', text)
    # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove single letters
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    # Remove pure numbers
    text = re.sub(r'\b\d+\b', '', text)
    # Remove common words that don't add meaning
    tech_words = {'contenttype', 'charset', 'contenttransferencoding', 'base64', 'attachment', 'filename'}
    text = ' '.join(word for word in text.split() if word not in tech_words)
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
    body_text = body_match.group(1).strip() if body_match else None
    body_text_cleaned = preprocess_body_text(body_match.group(1).strip()) if body_match else None

    return {
        'from': from_email,
        'to': to_email,
        'cc': cc_email,
        'subject': subject_email,
        'BodyText': body_text,
        'CleanedText': body_text_cleaned
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
    fraudolent_emails['body'] = fraudolent_emails['body']
    
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
        #tokens = [(t.get("word"), t.get("pos")) for t in post.findall(".//t")]  # Tokens and POS
        #word_count = len(tokens)  # Number of tokens (words) in the post

        # Append to data
        data.append({
            "File": os.path.basename(file_path),  # Name of the file
            "User": user,
            "Class": post_class,
            "Content": content
            #"Tokens": tokens,
            #"WordCount": word_count,
            #"FirstWord": tokens[0][0] if tokens else None,  # First word in the post
            #"LastWord": tokens[-1][0] if tokens else None,  # Last word in the post
        })

    return data

def preprocess_text_for_lda(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english') and len(word) > 2]
    # Lemmatize the tokens (skip for now)
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Filter short tokens
    #tokens = [word for word in tokens if len(word) > 2]
    # Join tokens back to a string 
    return " ".join(tokens)

def preprocess_text_for_ner(text):
    """
    Preprocesses text specifically for Named Entity Recognition.
    Maintains capitalization and most punctuation while removing noise.
    
    Args:
    text (str): The input text to be preprocessed
    
    Returns:
    str: The cleaned text
    """
    # Remove email attachment markers and related content
    text = re.sub(r'_secatt_.*?(?=\s|$)', '', text)
    
    # Remove content type declarations and encoding information
    text = re.sub(r'contenttype\s+\S+', '', text)
    text = re.sub(r'content[-_]?type\s*:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'content\s*type\s*:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'charset\S+', '', text)
    text = re.sub(r'charset\S+', '', text)
    text = re.sub(r'contenttransferencoding\s+\S+', '', text)
    text = re.sub(r'contentdisposition\s+\S+', '', text)
    
    # Remove base64 and other encoding markers
    text = re.sub(r'base64\s+\S+', '', text)
    text = re.sub(r'quotedprintable\s+', '', text)
    
    # Remove file attachments and names
    text = re.sub(r'filename\S+', '', text)
    text = re.sub(r'name\S+', '', text)
    
    # Remove encoded special characters (like 2e, 3a, etc.)
    text = text.replace('2e', '.').replace('3a', ':').replace('2c', ',').replace('40', '@')
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text