import os  # Imports the os module, which provides a way of interacting with the operating system (file handling, etc.)
import typer  # Imports the Typer library for creating command-line interfaces (CLI)
import string  # Imports the string module for common string operations (punctuation, etc.)
import numpy as np  # Imports the numpy library for numerical operations
import pandas as pd  # Imports the pandas library for data manipulation and analysis
import imaplib  # Imports the imaplib module to interact with email servers using the IMAP protocol
import email  # Imports the email module for parsing and creating email messages
from email.header import decode_header  # Imports the decode_header function for decoding email headers
from sklearn.feature_extraction.text import TfidfVectorizer  # Imports the TF-IDF vectorizer from scikit-learn
from sklearn.model_selection import train_test_split  # Imports train_test_split to split data into training and test sets
from sklearn.ensemble import RandomForestClassifier  # Imports the RandomForestClassifier for building classification models
import nltk  # Imports the nltk library for natural language processing (NLP)
from nltk.corpus import stopwords  # Imports the stopwords corpus from nltk
from nltk.stem.porter import PorterStemmer  # Imports the PorterStemmer class for stemming words
from transformers import pipeline  # Imports the Hugging Face pipeline for text generation
import torch  # Imports the torch library for machine learning and deep learning operations
import smtplib  # Imports smtplib to send emails using the Simple Mail Transfer Protocol (SMTP)
from email.mime.text import MIMEText  # Imports MIMEText to create email messages with text content
from email.mime.multipart import MIMEMultipart  # Imports MIMEMultipart for creating email messages with multiple parts
import schedule  # Imports the schedule library to schedule tasks in Python
import time  # Imports the time module to work with time-related tasks
import joblib  # Imports joblib for serializing and deserializing Python objects, particularly models
import logging  # Imports the logging module to log messages for debugging and tracking
from bs4 import BeautifulSoup  # Imports BeautifulSoup from the bs4 library for parsing HTML and XML
from dotenv import load_dotenv  # Imports load_dotenv to load environment variables from a .env file
from sklearn.metrics import classification_report, accuracy_score  # Imports methods for evaluating machine learning models
import re  # Imports the re module for regular expression operations
import os
# Optional: only use if you installed language_tool_python
try:
    USE_GRAMMAR_CORRECTION = True  # Tries to enable grammar correction if the library is installed
except ImportError:
    USE_GRAMMAR_CORRECTION = False  # If not installed, disables grammar correction

# Initializes the GPT-Neo model for text generation using the Hugging Face pipeline
generator = pipeline(
    'text-generation',  # Specifies the task (text generation)
    model='EleutherAI/gpt-neo-1.3B',  # Specifies the model (GPT-Neo 1.3B)
    device=0,  # Specifies to use GPU if available (device 0)
    torch_dtype=torch.float16  # Specifies the data type for PyTorch (float16 for lower memory usage)
)

# Loads environment variables from the .env file to keep sensitive information secure
load_dotenv()

# Initializes the Typer app for the command-line interface (CLI)
app = typer.Typer()

# Downloads NLTK stopwords (only once)
nltk.download('stopwords', quiet=True)

# Loads credentials and API keys from environment variables
EMAIL = os.getenv("EMAIL")  # Your email address
PASSWORD = os.getenv("PASSWORD")  # Your email password (should be kept secure)
OPENAI_API_KEY = os.getenv ("openai") # OpenAI API key for using GPT models

# Configures logging to output log messages at the INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to hold the trained model and vectorizer for spam detection
MODEL = None
VECTORIZER = None

# Function to preprocess text for spam detection
def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, stopwords, and stemming."""
    stemmer = PorterStemmer()  # Initializes a Porter Stemmer for stemming words
    stopwords_set = set(stopwords.words('english'))  # Loads the English stopwords from NLTK
    text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()  # Converts text to lowercase, removes punctuation, and splits into words
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]  # Stems words and removes stopwords
    return ' '.join(text)  # Joins the processed words back into a string

# Command to train the spam detection model
@app.command()
def train_model():
    """Train the spam detection model and save it to disk."""
    logger.info("📊 Training model...")  # Logs the start of model training

    try:
        df = pd.read_csv('spam_ham_dataset.csv')  # Loads the dataset containing spam and ham (non-spam) emails
        df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ''))  # Removes any carriage return/line breaks in the text
        df = df.dropna(subset=['label_num'])  # Drops any rows where the 'label_num' column is missing (e.g., missing labels)

        corpus = df['text'].apply(preprocess_text)  # Preprocesses the text using the preprocess_text function

        vectorizer = TfidfVectorizer()  # Initializes a TF-IDF vectorizer to convert text into numerical features
        X = vectorizer.fit_transform(corpus)  # Transforms the text into a sparse matrix of TF-IDF features
        y = df.label_num  # Labels for spam (1) or ham (0)

        # Splits the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
        clf = RandomForestClassifier(n_jobs=-1)  # Initializes a Random Forest classifier
        clf.fit(X_train, y_train)  # Trains the classifier on the training data

        # Saves the trained model and vectorizer to disk
        joblib.dump(clf, 'spam_model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

        logger.info("✅ Model training completed and saved to disk!")  # Logs a success message
    except Exception as e:
        logger.error(f"⚠️ Failed to train model: {e}")  # Logs any errors that occur during training
    y_pred = clf.predict(X_test)  # Makes predictions on the test set
    print("Accuracy:", accuracy_score(y_test, y_pred))  # Prints the accuracy of the model
    print(classification_report(y_test, y_pred))  # Prints the classification report showing precision, recall, and F1-score

# Command to load the trained model and vectorizer
@app.command()
def load_model():
    """Load the trained model and vectorizer from disk."""
    global MODEL, VECTORIZER
    try:
        MODEL = joblib.load('spam_model.pkl')  # Loads the trained spam detection model from disk
        VECTORIZER = joblib.load('vectorizer.pkl')  # Loads the vectorizer from disk
        logger.info("✅ Model and vectorizer loaded successfully!")  # Logs a success message
    except Exception as e:
        logger.error(f"⚠️ Failed to load model: {e}")  # Logs any errors that occur while loading the model

# Function to extract the body of the email
def extract_email_body(msg):
    """Extract the body of the email with better MIME handling."""
    body = ""
    if msg.is_multipart():  # Checks if the email has multiple parts (e.g., plain text and HTML)
        for part in msg.walk():
            content_type = part.get_content_type()  # Gets the content type of the part
            content_disposition = str(part.get("Content-Disposition", ""))  # Gets the content disposition (e.g., inline or attachment)
            if "attachment" in content_disposition:  # Skips attachments
                continue
            try:
                payload = part.get_payload(decode=True)  # Decodes the payload of the part
                if payload:
                    decoded = payload.decode("utf-8", errors="replace")  # Decodes the payload to a string
                    if content_type == "text/plain":  # If the part is plain text, return it
                        return decoded
                    elif content_type == "text/html" and not body.strip():  # If the part is HTML, return it if no text is found yet
                        body = BeautifulSoup(decoded, "html.parser").get_text()  # Parses the HTML and extracts the text
            except Exception as e:
                logger.warning(f"⚠️ Failed to decode part: {e}")  # Logs any errors that occur during decoding
    else:  # If the email is not multipart, attempt to decode it as a single part
        try:
            payload = msg.get_payload(decode=True)  # Decodes the payload
            if payload:
                body = payload.decode("utf-8", errors="replace")  # Decodes the payload to a string
        except Exception as e:
            logger.warning(f"⚠️ Failed to decode email: {e}")  # Logs any errors
    return body.strip()  # Returns the cleaned body of the email

# Command to check for new spam emails and respond
@app.command()
def check_email():
    """Fetch emails from the spam folder and classify them as spam or ham."""
    if MODEL is None or VECTORIZER is None:
        logger.warning("⚠️ Train the model first...")  # Logs a warning if the model is not loaded
        return

    logger.info("📬 Checking emails...")  # Logs the start of the email checking process
    mail = None

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")  # Connects to the Gmail IMAP server
        mail.login(EMAIL, PASSWORD)  # Logs in to the email account using the credentials
        mail.select("[Gmail]/Spam")  # Selects the "Spam" folder

        status, messages = mail.search(None, 'UNSEEN')  # Searches for unseen (unread) messages
        email_ids = messages[0].split()  # Gets the list of email IDs

        for email_id in email_ids:
            try:
                _, msg_data = mail.fetch(email_id, "(BODY.PEEK[])")  # Fetches the email content
                raw_email = msg_data[0][1]  # Gets the raw email data
                msg = email.message_from_bytes(raw_email)  # Parses the raw email data

                # Decode subject
                subject, encoding = decode_header(msg["Subject"])[0]  # Decodes the email subject
                subject = subject.decode(encoding or "utf-8") if isinstance(subject, bytes) else subject  # Decodes the subject if needed

                body = extract_email_body(msg)  # Extracts the email body

                if not body:
                    logger.warning("⚠️ Empty or unreadable email, skipping...")  # Skips emails with no readable body
                    continue

                logger.info(f"📩 From: {msg['From']}, Subject: {subject}")  # Logs the email sender and subject

                # Preprocess and classify the email body
                email_text = preprocess_text(body)  # Preprocesses the email body text
                X_email = VECTORIZER.transform([email_text]).toarray()  # Converts the email text into feature vectors using the loaded vectorizer
                prediction = MODEL.predict(X_email)  # Predicts whether the email is spam or ham

                if prediction == 1:  # If the email is spam
                    logger.info("🚨 Spam detected! Generating response...")  # Logs a message indicating spam detection
                    generate_spam_response(msg['From'], subject)  # Generates and sends a sarcastic spam response

                time.sleep(1)  # Waits for 1 second before processing the next email
            except Exception as e:
                logger.error(f"⚠️ Failed to process email {email_id}: {e}")  # Logs any errors during email processing
                if mail:
                    mail.store(email_id, '-FLAGS', '\\Seen')  # Marks the email as unread if processing fails
    except Exception as e:
        logger.error(f"⚠️ Critical error: {e}")  # Logs any critical errors during the email checking process
    finally:
        if mail:
            mail.logout()  # Logs out from the email server

# Command to generate a sarcastic response to a spam email
def post_process_gpt_output(raw_text, keywords=None):
    """
    Clean and structure GPT output into a readable, sarcastic scam-style email.
    """
    # 1. Trim to last complete sentence
    def trim_nonsense(text):
        sentences = re.split(r'(?<=[.!?]) +', text.strip())  # Splits the text into sentences
        if len(sentences) > 1:
            return ' '.join(sentences[:-1])  # Removes the last incomplete sentence
        return text

    # 2. Remove repeated lines
    def remove_repeats(text):
        lines = text.split('\n')  # Splits the text into lines
        seen = set()  # Set to track repeated lines
        result = []
        for line in lines:
            line_clean = line.strip()  # Strips whitespace from each line
            if line_clean and line_clean not in seen:  # Checks if the line is not a repeat
                result.append(line)  # Adds the line to the result
                seen.add(line_clean)  # Marks the line as seen
        return '\n'.join(result)  # Joins the lines back into a single string

    # 3. Check for required keywords
    def has_keywords(text, kw_list):
        if not kw_list:
            return True  # If no keywords are provided, always return True
        return all(kw.lower() in text.lower() for kw in kw_list)  # Checks if all keywords are present in the text

    # 4. Optional grammar fix
    def fix_grammar(text):
        return text  # Returns the text without any grammar fixes for now

    # 5. Wrap in scam email format
    def format_as_email(content):
        return f"""\
Subject: You've Been Had 😎

Dear Victim,

{content}

Sincerely,  
The Scammer Who Outsmarted You 💼
"""

    # --- Run all steps ---
    cleaned = trim_nonsense(raw_text)  # Trims the nonsense part from the raw GPT output
    no_repeats = remove_repeats(cleaned)  # Removes repeated lines from the output
    grammar_fixed = fix_grammar(no_repeats)  # Optionally fixes grammar (no-op here)
    
    if keywords and not has_keywords(grammar_fixed, keywords):  # Checks if all required keywords are present
        print("[!] Warning: Some keywords are missing. Output may be off-topic.")  # Prints a warning if keywords are missing

    return format_as_email(grammar_fixed)  # Formats the cleaned output as an email

# Command to generate and send a sarcastic response to a spam email
@app.command()
def generate_spam_response(to, subject):
    """Generate a sarcastic response to the given email and send it."""
    try:
        # Predefined sarcastic response prompt for GPT-3
        prompt = f"""\
        You have successfully scammed me! Congratulations on being so clever.
        Your email subject was: "{subject}"
        I’ve been waiting my whole life for such an offer!

        Here’s my personal info:
        [INSERT FAKE INFO HERE] — I’m sure you’ll use it wisely.

        Keep it up,
        The Gullible One 🏆
        """
        # Generate the email content using GPT-Neo
        response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)

        # Post-process the output to make it look like a genuine scam email
        response_text = post_process_gpt_output(response[0]['generated_text'])

        # Set up the email headers and body
        msg = MIMEMultipart()
        msg['From'] = EMAIL  # Sets the 'From' field in the email header
        msg['To'] = to  # Sets the 'To' field in the email header
        msg['Subject'] = "Re: " + subject  # Sets the email subject as a reply to the original email

        # Attach the response text to the email body
        msg.attach(MIMEText(response_text, 'plain'))

        # Send the email using SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Starts the TLS connection for secure email transmission
            server.login(EMAIL, PASSWORD)  # Logs in to the email server
            server.send_message(msg)  # Sends the email message

        logger.info(f"✅ Sent sarcastic response to {to}")  # Logs the success of sending the response
    except Exception as e:
        logger.error(f"⚠️ Failed to send response: {e}")  # Logs any errors that occur during sending the email
