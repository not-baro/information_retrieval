import pandas as pd
import re
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text_for_lda(text):
    """
    Preprocesses text for LDA topic modeling with comprehensive cleaning patterns.
    """
    # Expanded technical words and email-related terms
    tech_words = {
        'contenttype', 'contenttransferencoding', 'charset', 'base64', 
        'multipart', 'mime', 'quoted', 'printable', 'encoding', 'encoded',
        'attachment', 'boundary', 'content', 'type', 'transfer', 'plain', 'text',
        
        # Email client terms
        'subject', 'forwarded', 'message', 'original', 'mail', 'email', 'sent', 
        'received', 'from', 'to', 'cc', 'bcc', 'reply', 'forward',
        
        # Remove specific problematic terms we're seeing
        'qzsoftdirectmailseperator', 'ivmnmwqldoui', 'psmswqhdzsynwz', 
        'wyzcosnmxifdpi', 'multipart', 'mime',

        # Email client terms
        'subject', 'forwarded', 'message', 'original', 'mail', 'email', 'sent', 
        'attached', 'attachment', 'contact', 'click', 'internet', 'com', 'org', 'net',
        
        # Common email actions/status
        'forward', 'reply', 'sent', 'received', 'attached', 'attachment', 'copy',
        'download', 'upload', 'click', 'link', 'subscribe', 'unsubscribe',
        
        # Email formalities
        'dear', 'hello', 'hi', 'thanks', 'thank', 'regards', 'sincerely',
        'best', 'wishes', 'please', 'kindly', 'let', 'know', 'asap',
        
        # Previous technical terms
        'contenttype', 'charset', 'contenttransferencoding', 'base64', 
        'filename', 'npsb', 'fyi', 'thx', 'pls', 'plz', 'fwd', 're', 'fw',
        'attn', 'cc', 'bcc', 'ps', 'nb', 'ref', 'cdm', 'nt',
        
        # Meeting related
        'meeting', 'schedule', 'appointment', 'calendar', 'agenda',
        
        # Document related
        'document', 'file', 'pdf', 'doc', 'xls', 'attached', 'attachment',
        
        # Common business email terms
        'questions', 'information', 'need', 'use', 'agreement', 'deal',
        
        # Names of days and months
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december'
    }

    # Remove HTML and email attachments
    patterns_to_remove = [
        # Remove HTML and email attachments
        (r'<[^>]+>|_secatt_.*?(?=\s|$)', ''),
        
        # Remove email headers completely
        (r'(?i)^(?:subject|from|to|cc|bcc|sent|date|importance):\s*.*$\n?', '', re.MULTILINE),
        
        # Remove forwarded/replied message headers
        (r'(?i)(?:^|\n)[-]+\s*(?:forwarded|original)\s+message\s*[-]+.*?(?=\n\n|\Z)', '', re.DOTALL),
        
        # Remove email signatures
        (r'(?i)(?:^|\n)regards,.*?(?=\n\n|\Z)', '', re.DOTALL),
        (r'(?i)(?:^|\n)best,.*?(?=\n\n|\Z)', '', re.DOTALL),
        
        # Remove common email client artifacts
        (r'(?i)on.*wrote:', ''),
        (r'(?i)>+\s*', ''),
        
        # Remove URLs and email addresses more aggressively
        (r'(?i)https?://\S+', ''),
        (r'(?i)www\.\S+', ''),
        (r'(?i)\S+@\S+\.\S+', ''),
        (r'(?i)\.com\b|\.org\b|\.net\b', ''),
        
        # Remove attachment indicators
        (r'(?i)(?:attached|attachment|file):\s*\S+', ''),
        
        # Remove technical headers and encoding markers
        (r'content[-_]?type\s*:\s*\S+', '', re.IGNORECASE),
        (r'charset\S+', ''),
        (r'contenttransferencoding\s+\S+', ''),
        (r'contentdisposition\s+\S+', ''),
        (r'base64\s+\S+', ''),
        (r'quotedprintable\s+', ''),
        
        # Remove encoded characters and HTML entities
        (r'\b\d{3,4}[âãäåæ]\b', ''),
        (r'\b\d{4}[âãäåæ]\b', ''),
        (r'&#\d+;', ''),
        (r'&[a-z]+;', ''),
        
        # Remove long alphanumeric sequences
        (r'\b[a-zA-Z0-9]{20,}\b', ''),
        
        # Remove short alphanumeric combinations
        (r'\b\d[a-zA-Z]\b', ''),
        (r'\b[a-zA-Z]\d\b', ''),
        (r'\b\d{1,2}[a-zA-Z]\d{1,2}\b', ''),
        
        # Clean up remaining patterns
        (r'\s+', ' ')
    ]

    # Apply all cleaning patterns
    for pattern in patterns_to_remove:
        if len(pattern) == 2:
            text = re.sub(pattern[0], pattern[1], text)
        else:
            text = re.sub(pattern[0], pattern[1], text, flags=pattern[2])

    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [
        word for word in tokens 
        if word not in stop_words 
        and word.lower() not in tech_words  # Case insensitive check
        and len(word) > 2
        and not any(c.isdigit() for c in word)
        and not word.startswith('http')
    ]
    
    return " ".join(tokens)

def preprocess_body_text(text):
    """
    Preprocesses email body text by removing various unwanted elements.
    
    Args:
    text (str): The input text to be preprocessed
    
    Returns:
    str: The cleaned text
    """
    text = re.sub(r'<[^>]+>', '', text)
    # Remove email attachment markers and related content
    text = re.sub(r'_secatt_.*?(?=\s|$)', '', text)
    # Remove content type declarations and encoding information
    text = re.sub(r'content[-_]?type\s*:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'content\s*type\s*:\s*\S+', '', text, flags=re.IGNORECASE)
    tech_words = {'contenttype', 'charset', 'contenttransferencoding', 'base64', 'attachment', 'filename'}
    text = ' '.join(word for word in text.split() if word not in tech_words)
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
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text_for_ner(text):
    """
    Preprocesses text specifically for Named Entity Recognition.
    Preserves capitalization, punctuation, and text structure while removing technical artifacts.
    
    Args:
    text (str): The input text to be preprocessed
    
    Returns:
    str: The cleaned text maintaining original formatting
    """
    patterns_to_remove = [
        (r'<[^>]+>', ' '),  # Replace HTML with space to maintain sentence structure
        (r'_secatt_.*?(?=\s|$)', ' '),
        
        # Remove technical headers and encoding markers
        (r'content[-_]?type\s*:\s*\S+|charset\S+|contenttransferencoding\s+\S+|contentdisposition\s+\S+', 
         ' ', re.IGNORECASE),
        (r'base64\s+\S+|quotedprintable\s+|filename\S+', ' '),
        
        # Remove encoded characters and HTML entities while preserving structure
        (r'\b\d{3,4}[âãäåæ]\b', ' '),
        (r'\b\d{4}[âãäåæ]\b', ' '),
        (r'&#\d+;', ' '),
        (r'&[a-z]+;', ' '),
        (r'\b\d+[âãäåæ]\d+\b', ' '),
        (r'[âãäåæ]\s*\d+', ' '),
        
        # Clean specific technical patterns
        (r'(?:charset|encoding|content-type)=["\']?[\w-]+["\']?', ' '),
        
        # Remove obviously malformed email addresses while keeping valid ones
        (r'\S+@\S+\.\S+(?:\s+@\s+|\s+\.\s+)\S+', ' '),
        
        # Remove multiple spaces while preserving single newlines
        (r' +', ' '),
        (r'\n\n+', '\n\n')
    ]
    
    # Apply all cleaning patterns
    for pattern in patterns_to_remove:
        if len(pattern) == 2:
            text = re.sub(pattern[0], pattern[1], text)
        else:
            text = re.sub(pattern[0], pattern[1], text, flags=pattern[2])
    
    # Preserve sentence structure by ensuring proper spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)  # Add space after punctuation if missing
    
    return text.strip()

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
    """
    Preprocesses text for LDA topic modeling with comprehensive cleaning patterns.
    """
    # Expanded technical words and email-related terms
    tech_words = {
        'contenttype', 'contenttransferencoding', 'charset', 'base64', 
        'multipart', 'mime', 'quoted', 'printable', 'encoding', 'encoded',
        'attachment', 'boundary', 'content', 'type', 'transfer', 'plain', 'text',
        
        # Email client terms
        'subject', 'forwarded', 'message', 'original', 'mail', 'email', 'sent', 
        'received', 'from', 'to', 'cc', 'bcc', 'reply', 'forward',
        
        # Remove specific problematic terms we're seeing
        'qzsoftdirectmailseperator', 'ivmnmwqldoui', 'psmswqhdzsynwz', 
        'wyzcosnmxifdpi', 'multipart', 'mime',

        # Email client terms
        'subject', 'forwarded', 'message', 'original', 'mail', 'email', 'sent', 
        'attached', 'attachment', 'contact', 'click', 'internet', 'com', 'org', 'net',
        
        # Common email actions/status
        'forward', 'reply', 'sent', 'received', 'attached', 'attachment', 'copy',
        'download', 'upload', 'click', 'link', 'subscribe', 'unsubscribe',
        
        # Email formalities
        'dear', 'hello', 'hi', 'thanks', 'thank', 'regards', 'sincerely',
        'best', 'wishes', 'please', 'kindly', 'let', 'know', 'asap',
        
        # Previous technical terms
        'contenttype', 'charset', 'contenttransferencoding', 'base64', 
        'filename', 'npsb', 'fyi', 'thx', 'pls', 'plz', 'fwd', 're', 'fw',
        'attn', 'cc', 'bcc', 'ps', 'nb', 'ref', 'cdm', 'nt',
        
        # Meeting related
        'meeting', 'schedule', 'appointment', 'calendar', 'agenda',
        
        # Document related
        'document', 'file', 'pdf', 'doc', 'xls', 'attached', 'attachment',
        
        # Common business email terms
        'questions', 'information', 'need', 'use', 'agreement', 'deal',
        
        # Names of days and months
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june', 'july',
        'august', 'september', 'october', 'november', 'december'
    }

    # Remove HTML and email attachments
    patterns_to_remove = [
        # Remove HTML and email attachments
        (r'<[^>]+>|_secatt_.*?(?=\s|$)', ''),
        
        # Remove email headers completely
        (r'(?i)^(?:subject|from|to|cc|bcc|sent|date|importance):\s*.*$\n?', '', re.MULTILINE),
        
        # Remove forwarded/replied message headers
        (r'(?i)(?:^|\n)[-]+\s*(?:forwarded|original)\s+message\s*[-]+.*?(?=\n\n|\Z)', '', re.DOTALL),
        
        # Remove email signatures
        (r'(?i)(?:^|\n)regards,.*?(?=\n\n|\Z)', '', re.DOTALL),
        (r'(?i)(?:^|\n)best,.*?(?=\n\n|\Z)', '', re.DOTALL),
        
        # Remove common email client artifacts
        (r'(?i)on.*wrote:', ''),
        (r'(?i)>+\s*', ''),
        
        # Remove URLs and email addresses more aggressively
        (r'(?i)https?://\S+', ''),
        (r'(?i)www\.\S+', ''),
        (r'(?i)\S+@\S+\.\S+', ''),
        (r'(?i)\.com\b|\.org\b|\.net\b', ''),
        
        # Remove attachment indicators
        (r'(?i)(?:attached|attachment|file):\s*\S+', ''),
        
        # Remove technical headers and encoding markers
        (r'content[-_]?type\s*:\s*\S+', '', re.IGNORECASE),
        (r'charset\S+', ''),
        (r'contenttransferencoding\s+\S+', ''),
        (r'contentdisposition\s+\S+', ''),
        (r'base64\s+\S+', ''),
        (r'quotedprintable\s+', ''),
        
        # Remove encoded characters and HTML entities
        (r'\b\d{3,4}[âãäåæ]\b', ''),
        (r'\b\d{4}[âãäåæ]\b', ''),
        (r'&#\d+;', ''),
        (r'&[a-z]+;', ''),
        
        # Remove long alphanumeric sequences
        (r'\b[a-zA-Z0-9]{20,}\b', ''),
        
        # Remove short alphanumeric combinations
        (r'\b\d[a-zA-Z]\b', ''),
        (r'\b[a-zA-Z]\d\b', ''),
        (r'\b\d{1,2}[a-zA-Z]\d{1,2}\b', ''),
        
        # Clean up remaining patterns
        (r'\s+', ' ')
    ]

    # Apply all cleaning patterns
    for pattern in patterns_to_remove:
        if len(pattern) == 2:
            text = re.sub(pattern[0], pattern[1], text)
        else:
            text = re.sub(pattern[0], pattern[1], text, flags=pattern[2])

    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [
        word for word in tokens 
        if word not in stop_words 
        and word.lower() not in tech_words  # Case insensitive check
        and len(word) > 2
        and not any(c.isdigit() for c in word)
        and not word.startswith('http')
        and not word.endswith('ed')  # Remove past tense verbs
    ]
    
    return " ".join(tokens)

def preprocess_text_for_ner(text):
    """
    Preprocesses text specifically for Named Entity Recognition.
    Preserves capitalization, punctuation, and text structure while removing technical artifacts.
    
    Args:
    text (str): The input text to be preprocessed
    
    Returns:
    str: The cleaned text maintaining original formatting
    """
    patterns_to_remove = [
        # Remove HTML and email attachments while preserving structure
        (r'<[^>]+>', ' '),  # Replace HTML with space to maintain sentence structure
        (r'_secatt_.*?(?=\s|$)', ' '),
        
        # Remove technical headers and encoding markers
        (r'content[-_]?type\s*:\s*\S+|charset\S+|contenttransferencoding\s+\S+|contentdisposition\s+\S+', 
         ' ', re.IGNORECASE),
        (r'base64\s+\S+|quotedprintable\s+|filename\S+', ' '),
        
        # Remove encoded characters and HTML entities while preserving structure
        (r'\b\d{3,4}[âãäåæ]\b', ' '),
        (r'\b\d{4}[âãäåæ]\b', ' '),
        (r'&#\d+;', ' '),
        (r'&[a-z]+;', ' '),
        (r'\b\d+[âãäåæ]\d+\b', ' '),
        (r'[âãäåæ]\s*\d+', ' '),
        
        # Clean specific technical patterns
        (r'(?:charset|encoding|content-type)=["\']?[\w-]+["\']?', ' '),
        
        # Remove obviously malformed email addresses while keeping valid ones
        (r'\S+@\S+\.\S+(?:\s+@\s+|\s+\.\s+)\S+', ' '),
        
        # Remove multiple spaces while preserving single newlines
        (r' +', ' '),
        (r'\n\n+', '\n\n')
    ]
    
    # Apply all cleaning patterns
    for pattern in patterns_to_remove:
        if len(pattern) == 2:
            text = re.sub(pattern[0], pattern[1], text)
        else:
            text = re.sub(pattern[0], pattern[1], text, flags=pattern[2])
    
    # Preserve sentence structure by ensuring proper spacing around punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)  # Add space after punctuation if missing
    
    return text.strip()