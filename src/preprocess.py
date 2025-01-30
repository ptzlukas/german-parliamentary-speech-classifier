import pandas as pd
import re
import spacy
import logging
import yaml
from tqdm import tqdm
tqdm.pandas()  # for Progressbar

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load spaCy German model
try:
    nlp = spacy.load("de_core_news_sm")
except:
    logging.error("spaCy model 'de_core_news_sm' is not installed. Install it using: python -m spacy download de_core_news_sm")
    exit()

def load_stopwords(file_path):
    """
    Load custom stopwords from a text file.
    :param file_path: Path to the stopwords file (one word per line).
    :return: A set of stopwords.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        stopwords = set(line.strip().lower() for line in f)
    logging.info("Stopwords loaded from %s", file_path)
    return stopwords

def clean_text_for_topic(text, stopwords):
    """
    Clean and preprocess the text specifically for topic classification.
    :param text: Input text.
    :param stopwords: Set of stopwords to remove.
    :return: Cleaned and processed text for topic classification.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ not in ['DET', 'CCONJ', 'PRON', 'PART', 'ADP', 'AUX', 'INTJ', 'SYM']:
            if re.match(r"^[a-zA-ZäöüÄÖÜß]+$", token.text):
                lemma = token.lemma_.lower()
                if lemma not in stopwords:
                    tokens.append(lemma)
    return " ".join(tokens)

def clean_text_for_sentiment(text):
    """
    Clean and preprocess the text specifically for sentiment analysis.
    :param text: Input text.
    :return: Cleaned and processed text for sentiment analysis.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zäöüß ]', '', text)  # Remove non-alphabetic characters
    doc = nlp(text)  # Tokenize and lemmatize
    lemmatized_text = ' '.join([token.lemma_ for token in doc if not token.is_stop])  # Remove stopwords
    return lemmatized_text

def preprocess(params):
    """
    Perform preprocessing on the dataset and write separate files for topic modeling and sentiment analysis.
    :param params: Dictionary containing parameters for preprocessing.
    """
    input_path = params["preprocessing"]["input_path"]
    topic_output_path = params["preprocessing"]["topic_output_path"]
    sentiment_output_path = params["preprocessing"]["sentiment_output_path"]
    stopwords_path = params["preprocessing"]["stopwords_path"]

    # Load the dataset
    logging.info("Loading dataset from %s", input_path)
    data = pd.read_csv(input_path)

    if "speech" not in data.columns:
        logging.error("The input file must contain a 'speech' column.")
        raise ValueError("The input file must contain a 'speech' column.")

    # Load custom stopwords
    stopwords = load_stopwords(stopwords_path)

    # Create topic modeling dataset
    logging.info("Processing dataset for topic modeling...")
    topic_data = data.dropna(subset=["manuel_label_primary"])  # Keep only rows with topic labels
    topic_data["speech"] = topic_data["speech"].progress_apply(lambda x: clean_text_for_topic(x, stopwords))
    topic_data = topic_data.drop(columns=["manuel_sentiment"], errors="ignore")  # Remove speech_sentiment if it exists

    # Remove empty rows in the processed text
    topic_data = topic_data[topic_data["speech"].str.strip() != ""]
    topic_data.to_csv(topic_output_path, index=False)
    logging.info("Topic modeling dataset saved to %s", topic_output_path)

    # Create sentiment analysis dataset
    logging.info("Processing dataset for sentiment analysis...")
    sentiment_data = data.dropna(subset=["manuel_sentiment"])  # Keep only rows with sentiment labels
    sentiment_data["speech"] = sentiment_data["speech"].progress_apply(clean_text_for_sentiment)
    sentiment_data = sentiment_data.drop(columns=["manuel_label_primary", "manuel_label_secondary"], errors="ignore")  # Remove unnecessary columns

    # Remove empty rows in the processed text
    sentiment_data = sentiment_data[sentiment_data["speech"].str.strip() != ""]
    sentiment_data.to_csv(sentiment_output_path, index=False)
    logging.info("Sentiment analysis dataset saved to %s", sentiment_output_path)

if __name__ == "__main__":
    # Load parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    preprocess(params)
