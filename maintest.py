#maintest.py
import os
import typer
import string
import numpy as np
import pandas as pd
import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from transformers import pipeline
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import joblib
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.metrics import classification_report, accuracy_score
import re
from DeepAudio.model_loader_audio import AudioAnalyzer
import logger
import logging
from DeepImage.model_loader_img import ImageAnalyzer

generator = pipeline(
    'text-generation',
    model='EleutherAI/gpt-neo-1.3B',
    device=0,
    torch_dtype=torch.float16
)

# Load environment variables from .env file
load_dotenv()

# Initialize Typer app
app = typer.Typer()

# Download NLTK stopwords (only once)
nltk.download('stopwords', quiet=True)

#PASSWORD="stxe sisc qfnu zgqk"
# Load credentials from environment variables
#EMAIL = "arapnecmi2@gmail.com"
#PASSWORD =  "**** **** **** ****" 
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
OPENAI_API_KEY = os.getenv("openai")
HF_TOKEN = os.getenv("tokk")
# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and vectorizer
MODEL = None
VECTORIZER = None

# Preprocess text for spam detection
def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, stopwords, and stemming."""
    stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    return ' '.join(text)

# Debug için email sayısını kontrol etme
@app.command()
def debug_email_count():
    """Debug email counts in different folders"""
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, PASSWORD)
        
        # Check different folders
        folders = ["INBOX", "[Gmail]/Spam", "[Gmail]/All Mail"]
        for folder in folders:
            try:
                mail.select(folder)
                status, messages = mail.search(None, 'ALL')
                total_count = len(messages[0].split()) if messages[0] else 0
                
                status, unseen = mail.search(None, 'UNSEEN')
                unseen_count = len(unseen[0].split()) if unseen[0] else 0
                
                logger.info(f"📁 {folder}: {total_count} total, {unseen_count} unseen")
            except Exception as e:
                logger.warning(f"⚠️ Cannot access {folder}: {e}")
        
        mail.logout()
    except Exception as e:
        logger.error(f"⚠️ Debug failed: {e}")
# Train spam detection model
@app.command()
def train_model():
    """Train the spam detection model and save it to disk."""
    logger.info("📊 Training model...")

    try:
        df = pd.read_csv('spam_ham_dataset.csv')
        df['text'] = df['text'].apply(lambda x: x.replace('\r\n', ''))
        df = df.dropna(subset=['label_num'])

        corpus = df['text'].apply(preprocess_text)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        y = df.label_num

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_jobs=-1)
        clf.fit(X_train, y_train)

        # Save model and vectorizer to disk
        joblib.dump(clf, 'spam_model.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')

        logger.info("✅ Model training completed and saved to disk!")
    except Exception as e:
        logger.error(f"⚠️ Failed to train model: {e}")
    y_pred = clf.predict(X_test) #accuracy prediction test
    print("Accuracy:", accuracy_score(y_test, y_pred))# Print accuracy score
    print(classification_report(y_test, y_pred)) # Print classification report

 # Load model and vectorizer from disk
@app.command()
def load_model():
    """Load the trained model and vectorizer from disk."""
    global MODEL, VECTORIZER
    try:
        MODEL = joblib.load('spam_model.pkl')
        VECTORIZER = joblib.load('vectorizer.pkl')
        logger.info("✅ Model and vectorizer loaded successfully!")
    except Exception as e:
        logger.error(f"⚠️ Failed to load model: {e}")

# Fetch and classify emails
def extract_email_body(msg):
    """Extract the body of the email with better MIME handling."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in content_disposition:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    decoded = payload.decode("utf-8", errors="replace")
                    if content_type == "text/plain":
                        return decoded
                    elif content_type == "text/html" and not body.strip():
                        body = BeautifulSoup(decoded, "html.parser").get_text()
            except Exception as e:
                logger.warning(f"⚠️ Failed to decode part: {e}")
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"⚠️ Failed to decode email: {e}")
    return body.strip()



def generalized_phishing_rules(email_subject, email_body):
    score = 0
    body_lower = email_body.lower()
    

    # Rule 1: Generic greeting without a name
    if re.search(r"\b(dear|hi|hello)\b", body_lower) and not re.search(r"\b[a-z]+\s[a-z]+\b", email_body, re.I):
        score += 1

    # Rule 2: Urgency or threatening language
    urgency_phrases = re.findall(r"\b(urgent|immediate|action required|within \d+ hours|final notice|prompt attention|verify within)\b", body_lower)
    score += 2 * len(urgency_phrases)

    # Rule 3: Requests for sensitive actions
    sensitive_actions = re.findall(r"\b(update|verify|confirm|reactivate|login|reset)\b.*\b(account|information|details|payment)\b", body_lower)
    score += 2 * len(sensitive_actions)

    # Rule 4: Suspicious links
    suspicious_links = re.findall(r"(click here|http[s]?://\S+)", body_lower)
    score += 2 * len(suspicious_links)

    # Rule 5: Impersonation of known services
    brand_mentions = re.findall(r"\b(amazon|netflix|paypal|microsoft|bank|apple|dhl|ups|irs|revenue|support team)\b", body_lower)
    score += 1 * len(brand_mentions)

    # Rule 6: Threats of account closure or legal consequences
    threats = re.findall(r"\b(suspended|closed|blocked|legal action|lawsuit|fine|report|service interruption|restricted access|closure)\b", body_lower)
    score += 2 * len(threats)

    return score >= 8


@app.command()
def check_email():
    if MODEL is None or VECTORIZER is None:
        logger.warning("⚠️ Train the model first...")
        return

    logger.info("📬 Checking emails...")
    mail = None

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, PASSWORD)
        mail.select("[Gmail]/Spam")  # or "inbox"

        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()

        for email_id in email_ids:
            try:
                _, msg_data = mail.fetch(email_id, "(BODY.PEEK[])")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                # Decode subject
                subject, encoding = decode_header(msg["Subject"])[0]
                subject = subject.decode(encoding or "utf-8") if isinstance(subject, bytes) else subject
                logger.info(f"📩 From: {msg['From']}, Subject: {subject}")

                # Extract body and attachments
                body = extract_email_body(msg)
                audiolabel = "Real"
                imagelabel = "Real"
                # Analyze attachments 
                for part in msg.walk():
                    content_disposition = str(part.get("Content-Disposition", ""))
                    if "attachment" in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            decoded_filename, enc = decode_header(filename)[0]
                            if isinstance(decoded_filename, bytes):
                                decoded_filename = decoded_filename.decode(enc or 'utf-8')

                            # Check for image types
                            if any(decoded_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                                attachment_data = part.get_payload(decode=True)
                                if attachment_data:
                                    temp_path = f"./temp_{decoded_filename}"
                                    with open(temp_path, 'wb') as f:
                                        f.write(attachment_data)
                                    logger.info(f"📷 Image saved: {temp_path}")

                                    # Analyze the image
                                    
                                    imageflag=analyze_image(temp_path)
                                    if imageflag is not None:
                                       logger.info(f"Image analysis result: {imageflag}")    
                                       imagelabel = imageflag.get('Prediction', 'Unknown')  # Default to 'Unknown' if 'Prediction' is missing
                                    else:
                                       logger.warning("⚠️ Image analysis returned None, skipping image prediction.")
                                       imagelabel = 'Unknown'


                                    
                                    os.remove(temp_path)
                            #Check for audio types        
                            elif any(decoded_filename.lower().endswith(ext) for ext in ['.wav', '.mp3']):
                                attachment_data = part.get_payload(decode=True)
                                if attachment_data:
                                    temp_path = f"./temp_{decoded_filename}"
                                    with open(temp_path, 'wb') as f:
                                        f.write(attachment_data)
                                    logger.info(f"Audio Saved: {temp_path}")

                                    # Analyze the audio
                                    audioflag=predict_audio_cli(temp_path,model_path='Deepaudio/xgb_voice_model.pkl',
    verbose=False)
                                    # Check if audioflag is not None before accessing the 'prediction' key
                                    if audioflag is not None:
                                       audiolabel = audioflag.get('prediction', 'Real')  # Default to 'Real' if 'prediction' is missing
                                    else:
                                       logger.warning("⚠️ Audio analysis returned None, skipping audio prediction.")
                                       audiolabel = 'Real'
                                    os.remove(temp_path) 

                if not body:
                    logger.warning("⚠️ Empty or unreadable email, skipping...")
                    continue

                # Preprocess and classify
                email_text = preprocess_text(body)
                X_email = VECTORIZER.transform([email_text]).toarray()
                prediction = int(MODEL.predict(X_email)[0])


                phishing_detected = generalized_phishing_rules(subject, body)
                logger.info(f"🧪 Spam Condition Check:")
                logger.info(f"- prediction: {prediction} (type: {type(prediction)})")
                logger.info(f"- phishing_detected: {phishing_detected}")
                logger.info(f"- audiolabel: {audiolabel}")
                logger.info(f"- imagelabel: {imagelabel}")
                if prediction == 1 or phishing_detected or audiolabel == "Fake" or imagelabel == "Deepfake":
                    logger.info("🚨 Spam detected! Generating response...")
                    generate_spam_response(msg['From'], subject)

                time.sleep(1)
            except Exception as e:
                logger.error(f"⚠️ Failed to process email {email_id}: {e}")
                if mail:
                    mail.store(email_id, '-FLAGS', '\\Seen')
    except Exception as e:
        logger.error(f"⚠️ Critical error: {e}")
    finally:
        if mail:
            mail.logout()



def post_process_gpt_output(raw_text, keywords=None):
    """
    Clean and structure GPT output into a readable, scam-style email.
    """
    
    # 1. Trim to last complete sentence
    def trim_nonsense(text):
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        if len(sentences) > 1:
            return ' '.join(sentences[:-1])
        return text

    # 2. Remove repeated lines
    def remove_repeats(text):
        lines = text.split('\n')
        seen = set()
        result = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                result.append(line)
                seen.add(line_clean)
        return '\n'.join(result)

   

    
   
    # 3. Wrap in scam email format
    def format_as_email(content):
        return f"""\
Subject: You've Been Had 😎

Dear Victim,

{content}

Sincerely,  
The Scammer Who Outsmarted You 💼
"""

    # --- Run all steps ---
    cleaned = trim_nonsense(raw_text)
    no_repeats = remove_repeats(cleaned)
    
    
    

    return format_as_email(no_repeats)
@app.command()
def generate_spam_response(recipient, subject):
    """Generate a spam response email using GPT model."""
    try:
        prompt = f"Write a sarcastic response to a spam email with the subject '{subject}'."
        response = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        formatted_response = post_process_gpt_output(response)
        send_spam_email(recipient, formatted_response, subject)
    except Exception as e:
        logger.error(f"⚠️ Failed to generate spam response: {e}")

@app.command()
def send_spam_email(recipient, message,subject):
    """Send a response email to the spam sender."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = recipient
        msg['Subject'] = f"Re: {subject}"

        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.ehlo("localhost")
            server.login(EMAIL, PASSWORD)
            server.sendmail(EMAIL, recipient, msg.as_string())

        logger.info(f"📩 Spam response sent to {recipient}")
    except Exception as e:
        logger.error(f"⚠️ Failed to send email: {e}")



@app.command()
def check_balance():
    df = pd.read_csv('spam_ham_dataset.csv',encoding='latin1')
    print(df['label'].value_counts())
    print(df.columns)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_image(image_path):
    #Analyze the image for deepfake detection.
    analyzer=ImageAnalyzer()
    result = analyzer.predict(image_path)
    if result["Prediction"] == "error":
        logger.error(f"Error while predicting image: {result.get('error_message')}")
    else:
        logger.info(f"Prediction: {result['Prediction']}")
        logger.info(f"Possibilities: {result['Possibilities']}")
    return result


@app.command()
def predict_audio_cli(
    file_path: str = typer.Argument(..., help="Path to audio file (.wav format)"),
    model_path: str = typer.Option(
        r'Deepaudio\xgb_voice_model.pkl',
        help="Custom path to trained model file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed processing information")
):
    """
    Analyze an audio file to determine if it's real or synthetic (deepfake).
    
    """
    try:
        # Initialize with verbose logging if requested
        if verbose:
            typer.echo("🔍 Initializing audio analyzer...")
            typer.echo(f"📂 Model path: {model_path}")
            typer.echo(f"🎵 Audio file: {file_path}")
        
        # Verify audio file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at: {file_path}")
        
        if verbose:
            typer.echo("✅ File verification passed")
            typer.echo("🔄 Loading audio model...")
        
        analyzer = AudioAnalyzer(model_path)
        
        if verbose:
            typer.echo("🎶 Extracting audio features...")
        
        result = analyzer.predict(file_path)
        
        if "error" in result:
            raise RuntimeError(result["error"])
        
        # Format the output with colors
        if result["prediction"] == "Real":
            color = typer.colors.GREEN
            emoji = "✅"
        else:
            color = typer.colors.RED
            emoji = "❌"
        
        if verbose:
            typer.echo("📊 Prediction results:")
        
        if result["confidence"]:
            typer.secho(
                f"{emoji} {result['prediction']} (Confidence: {result['confidence']*100:.2f}%)",
                fg=color,
                bold=True
            )
        else:
            typer.secho(
                f"{emoji} {result['prediction']}",
                fg=color,
                bold=True
            )
            
    except Exception as e:
        typer.secho(f"🔥 Error: {str(e)}", fg=typer.colors.RED, err=True)
        if verbose:
            typer.echo("\n💡 Troubleshooting tips:")
            typer.echo("- Ensure the audio file is in .wav format")
            typer.echo("- Check the model file exists at the specified path")
            typer.echo("- Verify all dependencies are installed (librosa, numpy, etc.)")
        raise typer.Exit(code=1)
    return result;  

# Run the app
if __name__ == "__main__":
    load_model() #Load the model at the start of the app
    app()