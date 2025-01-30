import os
import logging
import re
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec  # For progress tracking during training
from nltk.corpus import stopwords
from tqdm import tqdm

# Configure logging to display timestamps, log level, and messages
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

class TqdmCallback(CallbackAny2Vec):
    """
    Gensim callback class for displaying training progress using tqdm.
    """
    def __init__(self, epochs):
        self.epochs = epochs
        self.pbar = None
        self.current_epoch = 0

    def on_train_begin(self, model):
        """
        Called at the beginning of training. Initializes the tqdm progress bar.
        """
        self.pbar = tqdm(total=self.epochs, desc="Training", unit="epoch")

    def on_epoch_end(self, model):
        """
        Called at the end of each epoch. Updates the progress bar by one step.
        """
        self.current_epoch += 1
        self.pbar.update(1)

    def on_train_end(self, model):
        """
        Called at the end of training. Closes the progress bar.
        """
        self.pbar.close()

def load_stopwords(stopwords_path):
    """
    Loads custom stopwords from a file.
    
    Args:
        stopwords_path (str): Path to the stopwords file.
    
    Returns:
        set: A set of stopwords.
    
    Raises:
        FileNotFoundError: If the stopwords file does not exist.
    """
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"Stopword file not found: {stopwords_path}")
    
    logging.info("Loading custom stopwords...")
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stop_words = set(line.strip() for line in file)
    logging.info(f"{len(stop_words)} stopwords loaded.")
    return stop_words

def load_data(input_path, text_column):
    """
    Loads text data from a CSV file and retrieves the specified text column.
    
    Args:
        input_path (str): Path to the input CSV file.
        text_column (str): Name of the column containing text data.
    
    Returns:
        list: A list of text strings.
    
    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logging.info("Loading dataset...")
    data = pd.read_csv(input_path)
    texts = data[text_column].dropna().tolist()
    logging.info(f"{len(texts)} texts loaded.")
    return texts

def preprocess_texts(texts, stop_words):
    """
    Tokenizes German texts and removes stopwords (no stemming applied).
    
    Args:
        texts (list): List of text strings to preprocess.
        stop_words (set): Set of stopwords to remove from the texts.
    
    Returns:
        list: List of tokenized and filtered texts.
    """
    logging.info("Tokenizing texts and removing stopwords...")
    tokenized_texts = []
    
    # Use tqdm to display a progress bar during text preprocessing
    for text in tqdm(texts, desc="Preprocessing", unit="text"):
        # Extract only alphabetic words and convert to lowercase
        tokens = re.findall(r'\b[a-zäöüß]+\b', text.lower())
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words]
        tokenized_texts.append(filtered_tokens)
    
    logging.info(f"Remaining words after stopword removal: {sum(len(tokens) for tokens in tokenized_texts)}")
    return tokenized_texts

def train_word2vec(tokenized_texts,
                   vector_size=100,
                   window=5,
                   min_count=1,
                   workers=4,
                   epochs=10,
                   output_path="word2vec.model"):
    """
    Trains a Word2Vec model and saves it to a file, displaying progress with tqdm.
    
    Args:
        tokenized_texts (list): List of tokenized texts for training.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between a target word and its neighbors.
        min_count (int): Ignores words with total frequency lower than this.
        workers (int): Number of worker threads for training.
        epochs (int): Number of training epochs.
        output_path (str): File path to save the trained model.
    """
    logging.info("Training Word2Vec model...")
    
    # Initialize and train the model with a tqdm-based callback for progress tracking
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        callbacks=[TqdmCallback(epochs)]  # Progress bar callback
    )

    logging.info("Saving the model...")
    model.save(output_path)
    logging.info(f"Model saved at: {output_path}")

def main():
    """
    Main function for training a Word2Vec model on German texts.
    """
    # File paths and parameters
    input_path = "data/final_processed_dataset.csv"       # Path to the input dataset
    text_column = "speech"                 # Column containing text data
    stopwords_path = "resources/stopwords.txt"  # Path to the stopwords file
    output_path = "models/word2vec_sm.model"    # Output path for the trained model

    # Word2Vec hyperparameters
    vector_size = 150
    window = 10
    min_count = 5
    workers = 4
    epochs = 25  # Number of training epochs

    # Execution steps
    stop_words = load_stopwords(stopwords_path)
    texts = load_data(input_path, text_column)
    tokenized_texts = preprocess_texts(texts, stop_words)
    
    train_word2vec(
        tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        output_path=output_path
    )

if __name__ == "__main__":
    main()
